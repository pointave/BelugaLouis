import sys
import os
import json
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QListWidget, QLineEdit, QLabel, QMessageBox, QComboBox, QProgressBar, QSplitter,
    QListWidgetItem, QCheckBox, QSlider, QMenu, QSizePolicy
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QSettings, QThread, QObject, pyqtSignal, pyqtSlot, QSize, QUrl
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import requests
import concurrent.futures
import queue
import threading
import numpy as np
import concurrent.futures
import queue
import threading

class FileProcessingWorker(QObject):
    """Worker class for processing files in a separate thread."""

    # Signals for communication with main thread
    progress_updated = pyqtSignal(int, int)  # current, total
    file_processed = pyqtSignal(int, str)    # file_index, new_name
    processing_finished = pyqtSignal(int)    # total_processed
    processing_stopped = pyqtSignal(int)     # total_processed
    error_occurred = pyqtSignal(str, str)    # filename, error_message

    def __init__(self, files, directory, model, case_func, system_prompt, rename_history, prefix, suffix, add_date, use_filename_mode=False, custom_instructions=""):
        super().__init__()
        self.files = files
        self.directory = directory
        self.model = model
        self.case_func = case_func
        self.system_prompt = system_prompt
        self.rename_history = rename_history
        self.prefix = prefix
        self.suffix = suffix
        self.add_date = add_date
        self.use_filename_mode = use_filename_mode
        self.custom_instructions = custom_instructions
        self.is_running = False

    def resize_image_for_api(self, image_data, max_size=800):
        """Resize image to reduce payload size while maintaining aspect ratio."""
        import numpy as np
        import base64
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return base64.b64encode(image_data).decode('utf-8')
        
        # Get dimensions
        height, width = img.shape[:2]
        
        # Only resize if larger than max_size
        if max(height, width) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode with lower quality for faster transfer
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buf = cv2.imencode('.jpg', img, encode_param)
        
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    @pyqtSlot()
    def process_files(self):
        """Process files in the background."""
        try:
            self.is_running = True
            processed_count = 0

            try:
                for i, filename in enumerate(self.files):
                    if not self.is_running:
                        self.processing_stopped.emit(processed_count)
                        return

                    src = os.path.join(self.directory, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']

                    # Emit progress update
                    self.progress_updated.emit(i, len(self.files))

                    try:
                        # Process the file
                        images_b64 = []
                        prompt_lines = []

                        if self.use_filename_mode:
                            # Filename mode - use filename with custom instructions
                            base_name = os.path.splitext(filename)[0]  # Remove extension
                            prompt_lines = [
                                f'Generate a new filename based on: "{base_name}"',
                                '',
                                f'Instructions: {self.custom_instructions}',
                                '',
                                f'Requirements:',
                                f'- Use {self.case_func.__name__ if hasattr(self.case_func, "__name__") else "custom case"}',
                                '- Maximum 20 characters',
                                '- English only',
                                '- NO file extension',
                                '- NO special characters',
                                '- Follow the specific instructions provided',
                                '',
                                'Respond with ONLY the filename, nothing else.'
                            ]
                            print(f"DEBUG: Filename mode - base_name: {base_name}, instructions: {self.custom_instructions}")
                        else:
                            # Vision mode - analyze images/videos
                            prompt_lines = [
                                'Generate a descriptive filename for this file.',
                                '',
                                f'Requirements:',
                                f'- Use {self.case_func.__name__ if hasattr(self.case_func, "__name__") else "custom case"}',
                                '- Maximum 20 characters',
                                '- English only',
                                '- NO file extension',
                                '- NO special characters',
                                '- Focus on key visual elements',
                                '- Prefer one word if possible',
                                '- Use noun-verb format if multiple words',
                                '',
                                'Respond with ONLY the filename, nothing else.'
                            ]

                            if is_video:
                                # Extract frames from video
                                cap = cv2.VideoCapture(src)
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                frame_idxs = [0]
                                if total_frames > 1:
                                    frame_idxs.append(total_frames // 2)
                                if total_frames > 2:
                                    frame_idxs.append(total_frames - 1)

                                import base64
                                for idx in frame_idxs:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                                    ret, frame = cap.read()
                                    if ret:
                                        # Resize and compress frame
                                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                                        _, buf = cv2.imencode('.jpg', frame, encode_param)
                                        img_bytes = buf.tobytes()
                                        
                                        # Further resize if needed
                                        resized_b64 = self.resize_image_for_api(img_bytes)
                                        images_b64.append(resized_b64)
                                cap.release()
                                
                                if images_b64:
                                    video_prompt = f"\nThis is a video with {total_frames} frames. I'm showing you {len(images_b64)} key frames."
                                    prompt_lines.append(video_prompt)
                            
                            elif is_image:
                                # Image file - resize before encoding
                                try:
                                    with open(src, 'rb') as f:
                                        image_data = f.read()
                                        resized_b64 = self.resize_image_for_api(image_data)
                                        images_b64.append(resized_b64)
                                    prompt_lines.append('\nAnalyze this image to generate the filename.')
                                except Exception as e:
                                    raise ValueError(f'Could not read image file: {e}')
                            else:
                                # Non-media file - use filename as context
                                prompt_lines.append(f'\nOriginal filename for context: {filename}')

                        prompt = '\n'.join(prompt_lines)

                        # Handle different API endpoints for Ollama vs LM Studio
                        if self.model.startswith("lmstudio::"):
                            # LM Studio API call
                            actual_model = self.model.replace("lmstudio::", "")
                            
                            # Build messages with images if available
                            user_content = []
                            
                            # Add text prompt
                            user_content.append({
                                "type": "text",
                                "text": prompt
                            })
                            
                            # Add images if present
                            for img_b64 in images_b64:
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}"
                                    }
                                })
                            
                            lm_payload = {
                                "model": actual_model,
                                "messages": [
                                    {"role": "system", "content": self.system_prompt or "You are an expert file and media renamer."},
                                    {"role": "user", "content": user_content if images_b64 else prompt}
                                ],
                                "max_tokens": 50,
                                "temperature": 0.7
                            }
                            
                            # Check stop flag before API call
                            if not self.is_running:
                                self.processing_stopped.emit(processed_count)
                                return
                            
                            resp = requests.post('http://localhost:1234/v1/chat/completions', json=lm_payload, timeout=60)
                            if not self.is_running:
                                self.processing_stopped.emit(processed_count)
                                return
                            resp.raise_for_status()
                            data = resp.json()
                            full_response = data['choices'][0]['message']['content'].strip()
                        
                        else:
                            # Ollama API call
                            payload = {
                                "model": self.model,
                                "prompt": prompt,
                                "images": images_b64,
                                "stream": False,
                                "system": self.system_prompt or None
                            }
                            
                            # Check stop flag before API call
                            if not self.is_running:
                                self.processing_stopped.emit(processed_count)
                                return
                            
                            resp = requests.post('http://localhost:11434/api/generate', json=payload, timeout=60, stream=True)
                            if not self.is_running:
                                self.processing_stopped.emit(processed_count)
                                return
                            resp.raise_for_status()

                            # Handle streaming JSON lines
                            responses = []
                            for line in resp.iter_lines():
                                if not line or not self.is_running:
                                    break
                                decoded = line.decode('utf-8')
                                try:
                                    result = json.loads(decoded)
                                    if 'response' in result:
                                        responses.append(result['response'])
                                except Exception:
                                    continue
                            full_response = ''.join(responses).strip()

                        # Clean up the response (remove quotes, extra whitespace, etc.)
                        full_response = full_response.strip('"\'` \n\r')
                        print(f"DEBUG: Raw API response: '{full_response}'")
                        
                        # Remove any accidentally included file extensions from AI response
                        if '.' in full_response:
                            full_response = full_response.split('.')[0]
                        
                        # Apply case conversion (now correctly on filename WITHOUT extension)
                        cased_name = self.case_func(full_response)
                        cased_name = cased_name[:20]  # Max 20 characters
                        print(f"DEBUG: Processed response - cased_name: '{cased_name}'")
                        
                        if not cased_name:
                            raise ValueError('AI response did not produce a valid filename')

                        # Apply prefix and suffix
                        final_name = cased_name
                        if self.prefix:
                            final_name = self.prefix + final_name
                        if self.suffix:
                            final_name = final_name + self.suffix
                        
                        # Add date if requested
                        if self.add_date:
                            from datetime import datetime
                            date_str = datetime.now().strftime('%Y%m%d')
                            final_name = f"{date_str}_{final_name}"

                        new_name = final_name + ext
                        dst = os.path.join(self.directory, new_name)
                        print(f"DEBUG: Attempting to rename '{src}' to '{dst}'")

                        # Handle naming conflicts
                        counter = 1
                        while os.path.exists(dst):
                            new_name = f"{final_name}_{counter}{ext}"
                            dst = os.path.join(self.directory, new_name)
                            counter += 1

                        # Check stop flag before file operations
                        if not self.is_running:
                            self.processing_stopped.emit(processed_count)
                            return

                        # Rename the file
                        print(f"DEBUG: Renaming file from '{src}' to '{dst}'")
                        os.rename(src, dst)
                        processed_count += 1
                        print(f"DEBUG: Successfully renamed file, processed_count: {processed_count}")

                        # Emit file processed signal with index
                        self.file_processed.emit(i, new_name)

                    except Exception as e:
                        # Emit error signal
                        self.error_occurred.emit(filename, str(e))

            finally:
                if self.is_running:
                    self.processing_finished.emit(processed_count)

        except Exception as e:
            try:
                self.error_occurred.emit("worker", f"Critical worker error: {e}")
            except:
                pass

class ThumbnailListItem(QListWidgetItem):
    """Custom list item that can hold a thumbnail."""
    def __init__(self, text, thumbnail_path=None):
        super().__init__(text)
        self.thumbnail_path = thumbnail_path

class FileRenamer(QWidget):
    def __init__(self):
        # Initialize parent class first
        super().__init__()
        
        # Basic window setup
        self.setWindowTitle('Beluga Louis')
        self.resize(800, 500)
        
        # Initialize instance variables
        self.directory = ''
        self.files = []
        self.models = []
        self.is_processing = False
        
        # Initialize settings
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
        self.settings = {}
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.settings = {}
            
        # Load settings values
        self.last_selected_model = self.settings.get('last_model', '')
        self.last_opened_folder = self.settings.get('last_folder', '')
        self.rename_history = self.settings.get('rename_history', {})
        self.last_case = self.settings.get('last_case', 'kebab-case')
        self.system_prompt = self.settings.get('system_prompt', 'You are an expert file and media renamer. Always return a single, concise filename as instructed.')
        
        # Threading components
        self.worker_thread = None
        self.worker = None

        # Track original file order for real-time updates
        self.original_files_order = []
        self.cases = [
            ('kebab-case', 'kebab-case'),
            ('snake_case', 'snake_case'),
            ('camelCase', 'camelCase'),
            ('PascalCase', 'PascalCase'),
            ('Title Case', 'Title Case'),
            ('lower case', 'lower case'),
            ('UPPER CASE', 'UPPER CASE'),
        ]
        
        # Thumbnail cache
        self.thumbnail_cache = {}
        # Background thumbnail generation
        self._thumb_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._thumb_results = queue.Queue()
        self._thumb_pending = set()
        self.file_items = {}  # map file_path -> QListWidgetItem

        # Poll timer to process thumbnails from background threads
        from PyQt6.QtCore import QTimer
        self._thumb_timer = QTimer()
        self._thumb_timer.setInterval(80)
        self._thumb_timer.timeout.connect(self._process_thumbnail_results)
        self._thumb_timer.start()

        # Debounce timer for preview generation to avoid blocking during fast navigation
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._do_show_pending_preview)
        self._pending_preview_filename = None
        
        # Video player components
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.setMinimumSize(300, 300)

        # Mute audio and connect media player state changes
        self.audio_output.setVolume(0)  # Mute audio
        self.media_player.playbackStateChanged.connect(self.update_play_pause_button)
        self.media_player.positionChanged.connect(self.update_position_slider)
        self.media_player.durationChanged.connect(self.update_duration)
        
        # Track playback state for slider interaction
        self.was_playing = False
        
        # Set window icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'favicon.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print(f"DEBUG: Loaded icon from {icon_path}")
            else:
                # Fallback: try current directory
                icon_path = 'favicon.ico'
                if os.path.exists(icon_path):
                    self.setWindowIcon(QIcon(icon_path))
                    print(f"DEBUG: Loaded icon from {icon_path}")
                else:
                    print("DEBUG: Icon file not found")
        except Exception as e:
            print(f"DEBUG: Error loading icon: {e}")

        print("DEBUG: Initializing UI...")
        self.init_ui()
        print("DEBUG: UI initialized, showing window...")

    def toggle_renaming_mode(self):
        """Toggle between vision mode and filename mode."""
        if self.vision_mode_btn.isChecked():
            # Vision mode
            self.vision_mode_btn.setText('🔍 Vision Mode')
            self.custom_instructions_input.hide()
            self.info_label.setText('Files will be renamed using AI analysis of images/videos.')
        else:
            # Filename mode
            self.vision_mode_btn.setText('📝 Filename Mode')
            self.custom_instructions_input.show()
            self.info_label.setText('Files will be renamed using AI analysis of filenames with your custom instructions.')

    def init_ui(self):
        layout = QVBoxLayout()

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel('No folder selected')
        dir_btn = QPushButton('Select Folder')
        dir_btn.clicked.connect(self.select_folder)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(dir_btn)
        layout.addLayout(dir_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('AI Model:'))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.save_last_model)
        model_layout.addWidget(self.model_combo)
        load_btn = QPushButton('Load Models')
        load_btn.clicked.connect(self.load_models)
        model_layout.addWidget(load_btn)

        # Renaming mode toggle
        self.vision_mode_btn = QPushButton('🔍 Vision Mode')
        self.vision_mode_btn.setCheckable(True)
        self.vision_mode_btn.setChecked(True)  # Default to vision mode
        self.vision_mode_btn.clicked.connect(self.toggle_renaming_mode)
        model_layout.addWidget(self.vision_mode_btn)

        layout.addLayout(model_layout)

        # Case selection
        case_layout = QHBoxLayout()
        case_layout.addWidget(QLabel('Case:'))
        self.case_buttons = []
        from PyQt6.QtWidgets import QButtonGroup, QRadioButton
        self.case_group = QButtonGroup()
        for idx, (label, value) in enumerate(self.cases):
            btn = QRadioButton(label)
            self.case_group.addButton(btn, idx)
            case_layout.addWidget(btn)
            self.case_buttons.append(btn)
            if value == self.last_case:
                btn.setChecked(True)
        self.case_group.buttonClicked.connect(self.save_last_case)
        layout.addLayout(case_layout)

        # Naming options (prefix/suffix/date)
        naming_layout = QHBoxLayout()
        
        naming_layout.addWidget(QLabel('Prefix:'))
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText('Optional prefix')
        self.prefix_input.setMaxLength(10)
        naming_layout.addWidget(self.prefix_input)
        
        naming_layout.addWidget(QLabel('Suffix:'))
        self.suffix_input = QLineEdit()
        self.suffix_input.setPlaceholderText('Optional suffix')
        self.suffix_input.setMaxLength(10)
        naming_layout.addWidget(self.suffix_input)
        
        self.date_checkbox = QCheckBox('Add Date (YYYYMMDD)')
        naming_layout.addWidget(self.date_checkbox)
        
        layout.addLayout(naming_layout)

        # File list and preview splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - File list with thumbnails
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel('Files to rename:'))
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(48, 48))  # Set icon size for thumbnails
        self.file_list.itemSelectionChanged.connect(self.on_file_selection_changed)

        # Add context menu to file list
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)

        left_layout.addWidget(self.file_list)
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        # Right side - Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel('Preview:'))

        # Preview container that can switch between image and video
        self.preview_container = QWidget()
        self.preview_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_container_layout = QVBoxLayout(self.preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for more space

        # Image preview label (for images)
        self.preview_label = QLabel('')
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(100, 100)  # Smaller minimum size
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                qproperty-alignment: AlignCenter;
            }
        """)

        # Video preview widget (for videos)
        self.video_widget.setStyleSheet("QVideoWidget { background-color: #000000; }")
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.hide()  # Initially hidden

        preview_container_layout.addWidget(self.preview_label)
        preview_container_layout.addWidget(self.video_widget)

        right_layout.addWidget(self.preview_container)

        # Video controls (initially hidden)
        self.video_controls_layout = QHBoxLayout()

        self.play_pause_btn = QPushButton('▶️ Play')
        self.play_pause_btn.clicked.connect(self.toggle_video_playback)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.hide()  # Initially hidden
        self.video_controls_layout.addWidget(self.play_pause_btn)

        self.stop_btn = QPushButton('⏹️ Stop')
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        self.stop_btn.hide()  # Initially hidden
        self.video_controls_layout.addWidget(self.stop_btn)

        # Position slider for seeking
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)  # 0 to 1000 for percentage
        self.position_slider.setFixedHeight(20)  # Make it taller for easier grabbing
        self.position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #cccccc;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #666666;
                border: 1px solid #555555;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #777777;
            }
        """)
        self.position_slider.sliderMoved.connect(self.seek_video)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.hide()  # Initially hidden
        self.video_controls_layout.addWidget(self.position_slider)

        self.video_controls_layout.addStretch()
        right_layout.addLayout(self.video_controls_layout)

        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        splitter.setSizes([350, 550])  # Give more space to preview (left: file list, right: preview)
        layout.addWidget(splitter, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Restore last opened folder
        if self.last_opened_folder and os.path.isdir(self.last_opened_folder):
            self.directory = self.last_opened_folder
            self.dir_label.setText(self.last_opened_folder)
            self.refresh_file_list()

        # System prompt field
        sys_layout = QHBoxLayout()
        sys_layout.addWidget(QLabel('System Prompt:'))
        self.system_input = QLineEdit()
        self.system_input.setText(self.system_prompt)
        self.system_input.editingFinished.connect(self.save_system_prompt)
        sys_layout.addWidget(self.system_input)
        layout.addLayout(sys_layout)

        # Custom instructions for filename mode (initially hidden)
        self.custom_instructions_layout = QHBoxLayout()
        self.custom_instructions_layout.addWidget(QLabel('Custom Instructions:'))
        self.custom_instructions_input = QLineEdit()
        self.custom_instructions_input.setPlaceholderText('e.g., "remove the date", "rename in Spanish", "make it shorter"')
        self.custom_instructions_input.hide()  # Initially hidden
        self.custom_instructions_layout.addWidget(self.custom_instructions_input)
        layout.addLayout(self.custom_instructions_layout)

        # Info label
        self.info_label = QLabel('Files will be renamed using AI analysis of images/videos.')
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.rename_btn = QPushButton('Rename Files')
        self.rename_btn.clicked.connect(self.rename_files)
        buttons_layout.addWidget(self.rename_btn)
        
        revert_btn = QPushButton('Revert Renames')
        revert_btn.clicked.connect(self.revert_files)
        revert_btn.setEnabled(False)
        buttons_layout.addWidget(revert_btn)
        
        stop_btn = QPushButton('Stop Processing')
        stop_btn.clicked.connect(self.stop_processing)
        stop_btn.setEnabled(False)
        buttons_layout.addWidget(stop_btn)
        
        self.revert_button = revert_btn
        self.stop_button = stop_btn
        layout.addLayout(buttons_layout)

        self.setLayout(layout)
        self.load_models()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for video seeking and file navigation."""
        # Handle arrow keys for file list navigation (always available)
        if event.key() == Qt.Key.Key_Up:
            current_row = self.file_list.currentRow()
            if current_row > 0:
                self.file_list.setCurrentRow(current_row - 1)
            return
        elif event.key() == Qt.Key.Key_Down:
            current_row = self.file_list.currentRow()
            if current_row < self.file_list.count() - 1:
                self.file_list.setCurrentRow(current_row + 1)
            return

        # Handle video seek keys when video widget has focus
        if self.video_widget.hasFocus():
            duration = self.media_player.duration()
            if duration > 0:
                # Comma key - seek back 10 seconds
                if event.key() == Qt.Key.Key_Comma:
                    new_pos = max(0, self.media_player.position() - 10000)  # -10 seconds
                    self.media_player.setPosition(new_pos)
                    return

                # 0 and apostrophe keys - seek to beginning (0%)
                elif event.key() in [Qt.Key.Key_0, Qt.Key.Key_Apostrophe]:
                    new_pos = 0  # Beginning of video
                    self.media_player.setPosition(new_pos)
                    return

                # Number keys 1-9 - seek to percentage positions
                elif event.key() in [Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3,
                                   Qt.Key.Key_4, Qt.Key.Key_5, Qt.Key.Key_6,
                                   Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9]:
                    # Map number keys to positions (1 = 10%, 2 = 20%, ..., 9 = 90%)
                    percentage_map = {
                        Qt.Key.Key_1: 0.1, Qt.Key.Key_2: 0.2, Qt.Key.Key_3: 0.3,
                        Qt.Key.Key_4: 0.4, Qt.Key.Key_5: 0.5, Qt.Key.Key_6: 0.6,
                        Qt.Key.Key_7: 0.7, Qt.Key.Key_8: 0.8, Qt.Key.Key_9: 0.9
                    }
                    percentage = percentage_map[event.key()]
                    new_pos = int(duration * percentage)
                    self.media_player.setPosition(new_pos)
                    return

        super().keyPressEvent(event)

    def create_thumbnail(self, file_path):
        """Create a thumbnail for the file."""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
                # Image thumbnail
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    return pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                # Video thumbnail
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channels = frame_rgb.shape
                        bytes_per_line = channels * width
                        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        cap.release()
                        return pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    cap.release()
        except Exception:
            pass
        
        return None

    def _generate_thumbnail_bytes(self, file_path):
        """Background-friendly thumbnail generator. Returns JPEG bytes or None."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
                # Read image and resize with OpenCV
                with open(file_path, 'rb') as f:
                    data = f.read()
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > 48:
                    scale = 48.0 / max_dim
                    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    return buf.tobytes()
                return None
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return None
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    return None
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    return buf.tobytes()
                return None
        except Exception:
            return None

    def _process_thumbnail_results(self):
        """Process completed thumbnail bytes from background threads and set icons on items."""
        try:
            while True:
                file_path, data = self._thumb_results.get_nowait()
                try:
                    if data:
                        pix = QPixmap()
                        if pix.loadFromData(data):
                            icon = QIcon(pix)
                            self.thumbnail_cache[file_path] = icon
                            item = self.file_items.get(file_path)
                            if item:
                                item.setIcon(icon)
                except Exception:
                    pass
                finally:
                    self._thumb_pending.discard(file_path)
        except queue.Empty:
            return

    def _do_show_pending_preview(self):
        """Called by debounce timer to actually show the pending preview."""
        try:
            if self._pending_preview_filename:
                # It's safe to call the synchronous preview now; user likely paused navigation
                self.show_preview(self._pending_preview_filename)
        finally:
            self._pending_preview_filename = None

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
        return {}

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump({
                    'last_model': self.last_selected_model,
                    'last_folder': self.last_opened_folder,
                    'rename_history': self.rename_history,
                    'last_case': self.last_case,
                    'system_prompt': self.system_prompt
                }, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder', self.last_opened_folder)
        if folder:
            self.directory = folder
            self.dir_label.setText(folder)
            self.last_opened_folder = folder
            self.settings['last_folder'] = folder
            self.save_settings()
            self.revert_button.setEnabled(False)
            self.original_files_order = []
            self.thumbnail_cache.clear()  # Clear cache when changing folders
            self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list and store original order for real-time updates."""
        # Only include video and image files
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
        supported_exts = video_exts | image_exts

        current_files = [
            f for f in os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, f)) and
            os.path.splitext(f)[1].lower() in supported_exts
        ]
        self.files = current_files
        self.original_files_order = [(i, filename) for i, filename in enumerate(current_files)]
        self.file_list.clear()
        
        # Add files and schedule background thumbnail generation
        for filename in self.files:
            file_path = os.path.join(self.directory, filename)
            item = QListWidgetItem(filename)
            # Keep mapping from path to item for later icon updates
            self.file_items[file_path] = item

            # If thumbnail is cached, set it immediately
            if file_path in self.thumbnail_cache:
                item.setIcon(self.thumbnail_cache[file_path])
            else:
                # Schedule background thumbnail generation if not already pending
                if file_path not in self._thumb_pending:
                    self._thumb_pending.add(file_path)
                    future = self._thumb_executor.submit(self._generate_thumbnail_bytes, file_path)
                    # When done, put result into queue for main-thread processing
                    future.add_done_callback(lambda f, fp=file_path: self._thumb_results.put((fp, f.result())))

            self.file_list.addItem(item)
        
        self.clear_preview()

    def show_file_context_menu(self, position):
        """Show context menu for file list items."""
        item = self.file_list.itemAt(position)
        if item is None:
            return

        # Get the displayed filename (current name, not original)
        current_filename = item.text()

        # Create context menu
        context_menu = QMenu(self)

        # Add rename action
        rename_action = QAction("🔄 Rename this file", self)
        rename_action.triggered.connect(lambda: self.rename_single_file(current_filename))
        context_menu.addAction(rename_action)

        # Show the context menu at the cursor position
        context_menu.exec(self.file_list.mapToGlobal(position))

    def rename_single_file(self, filename):
        """Rename a single file using AI."""
        if not filename or not self.directory:
            return

        model = self.model_combo.currentText()
        if not model:
            QMessageBox.warning(self, 'No Model', 'Please select an AI model first.')
            return

        # Check if in filename mode and custom instructions are provided
        use_filename_mode = not self.vision_mode_btn.isChecked()
        if use_filename_mode:
            custom_instructions = self.custom_instructions_input.text().strip()
            if not custom_instructions:
                QMessageBox.warning(self, 'Missing Instructions', 'Please provide custom instructions for filename-based renaming.')
                return

        case_func = {
            'kebab-case': self.to_kebab_case,
            'snake_case': self.to_snake_case,
            'camelCase': self.to_camel_case,
            'PascalCase': self.to_pascal_case,
            'Title Case': self.to_title_case,
            'lower case': self.to_lower_case,
            'UPPER CASE': self.to_upper_case
        }[self.last_case]

        # Show progress
        self.rename_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

        # Stop any playing video to release file locks before renaming
        self.stop_video_for_rename()

        # Process single file in background thread
        files = [filename]
        use_filename_mode = not self.vision_mode_btn.isChecked()
        custom_instructions = self.custom_instructions_input.text().strip() if use_filename_mode else ""

        worker = FileProcessingWorker(
            files, self.directory, model, case_func,
            self.system_input.text().strip(), {},  # No rename history for single file
            "", "", False, use_filename_mode, custom_instructions  # No prefix/suffix/date for single file
        )

        worker_thread = QThread()
        worker.moveToThread(worker_thread)

        # Connect signals
        worker.progress_updated.connect(lambda current, total: self.progress_bar.setValue(current + 1))
        worker.file_processed.connect(self.on_single_file_processed)
        worker.processing_finished.connect(lambda total: self.on_single_file_finished(worker_thread, worker))
        worker.error_occurred.connect(self.on_error_occurred)

        worker_thread.started.connect(worker.process_files)
        worker_thread.start()

    def on_single_file_processed(self, file_index, new_name):
        """Handle single file processing completion."""
        # Update the file list item in real-time
        if file_index < self.file_list.count():
            current_item = self.file_list.item(file_index)
            if current_item:
                current_item.setText(new_name)

        # Update thumbnail cache key - use new_name for old_path since file has been renamed
        old_path = os.path.join(self.directory, new_name)  # File has already been renamed
        if old_path in self.thumbnail_cache:
            # Actually, we don't need to update cache here since we're not changing the filename in self.files
            pass

        # Update files list to maintain consistency for single file operations
        if file_index < len(self.files):
            self.files[file_index] = new_name

    def on_single_file_finished(self, worker_thread, worker):
        """Handle single file processing completion."""
        # Clean up thread
        worker_thread.quit()
        worker_thread.wait()

        # Hide progress bar and re-enable button
        self.progress_bar.setVisible(False)
        self.rename_btn.setEnabled(True)

        # Show success message
        QMessageBox.information(self, 'Success', 'File renamed successfully!')

    def stop_video_for_rename(self):
        """Safely stop video playback and release file locks before renaming."""
        try:
            # Stop media player if it's playing
            if self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
                self.media_player.stop()

            # Wait a moment for file locks to be released
            import time
            time.sleep(0.1)  # Brief pause to ensure file is released

            # Clear any cached video data
            if hasattr(self.media_player, 'setSource'):
                self.media_player.setSource(QUrl())

        except Exception as e:
            print(f"Warning: Could not fully stop video: {e}")

    def update_file_item_realtime(self, file_index, new_name):
        """Update a specific file item in the list with its new name while preserving position."""
        try:
            if file_index < self.file_list.count():
                current_item = self.file_list.item(file_index)
                if current_item:
                    current_item.setText(new_name)
                    # Update thumbnail cache key
                    old_path = os.path.join(self.directory, self.files[file_index])
                    new_path = os.path.join(self.directory, new_name)
                    if old_path in self.thumbnail_cache:
                        self.thumbnail_cache[new_path] = self.thumbnail_cache.pop(old_path)
        except Exception:
            pass

    def on_file_selection_changed(self):
        """Handle file selection changes to update preview."""
        selected_items = self.file_list.selectedItems()
        if selected_items:
            filename = selected_items[0].text()
            # Debounce heavy preview work so quick up/down navigation doesn't block UI
            self._pending_preview_filename = filename
            self._preview_timer.start()
        else:
            # Cancel any pending preview
            try:
                self._preview_timer.stop()
            except Exception:
                pass
            self._pending_preview_filename = None
            self.clear_preview()

    def show_preview(self, filename):
        """Show preview for the selected file."""
        if not filename or not self.directory:
            self.clear_preview()
            return

        file_path = os.path.join(self.directory, filename)

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                self.show_video_thumbnail(file_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
                self.show_image_preview(file_path)
            else:
                self.preview_label.setText(f'Preview not available for {ext.upper()} files')
        except Exception as e:
            self.preview_label.setText(f'Error loading preview: {str(e)}')

    def show_image_preview(self, file_path):
        """Show image preview."""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setText('Failed to load image')
        except Exception as e:
            self.preview_label.setText(f'Error loading image: {str(e)}')

    def show_video_thumbnail(self, file_path):
        """Extract and show video thumbnail."""
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channels = frame_rgb.shape
                    bytes_per_line = channels * width
                    q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)

                    # Get the available size for the preview
                    available_size = self.preview_container.size()
                    if available_size.width() <= 0 or available_size.height() <= 0:
                        available_size = QSize(400, 300)  # Fallback size

                    scaled_pixmap = pixmap.scaled(
                        available_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.preview_label.setPixmap(scaled_pixmap)
                else:
                    self.preview_label.setText('Failed to extract video frame')
            else:
                self.preview_label.setText('Failed to open video file')
            cap.release()
        except Exception as e:
            self.preview_label.setText(f'Error loading video: {str(e)}')

    def clear_preview(self):
        """Clear the preview area."""
        # Hide video controls and stop video
        self.play_pause_btn.hide()
        self.stop_btn.hide()
        self.position_slider.hide()
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.media_player.stop()

        # Show image preview label and hide video widget
        self.preview_label.show()
        self.video_widget.hide()
        self.preview_label.clear()
        self.preview_label.setText('Select a file to preview')

    def toggle_video_playback(self):
        """Toggle between play and pause for video playback."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_video(self):
        """Stop video playback."""
        self.media_player.stop()

    def update_play_pause_button(self, state):
        """Update play/pause button text based on media player state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_btn.setText('⏸️ Pause')
        else:
            self.play_pause_btn.setText('▶️ Play')

    def show_video_preview(self, file_path):
        """Show video preview with playback controls."""
        # Stop any currently playing video
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.media_player.stop()

        # Set up video source
        video_url = QUrl.fromLocalFile(file_path)
        self.media_player.setSource(video_url)

        # Show video widget and hide image preview
        self.video_widget.show()
        self.preview_label.hide()

        # Show video controls
        self.play_pause_btn.show()
        self.play_pause_btn.setEnabled(True)
        self.stop_btn.show()
        self.stop_btn.setEnabled(True)
        self.position_slider.show()

        # Auto-play the video
        self.media_player.play()

    def update_position_slider(self, position):
        """Update position slider when video position changes."""
        # Only update slider if user is not currently dragging it
        if not self.position_slider.isSliderDown():
            if self.media_player.duration() > 0:
                slider_pos = int((position / self.media_player.duration()) * 1000)
                self.position_slider.setValue(slider_pos)

    def update_duration(self, duration):
        """Update position slider range when duration changes."""
        if duration > 0:
            self.position_slider.setRange(0, 1000)

    def seek_video(self, value):
        """Seek video to position based on slider value."""
        if self.media_player.duration() > 0:
            position = int((value / 1000.0) * self.media_player.duration())
            self.media_player.setPosition(position)
            # Update the play/pause button if needed
            self.update_play_pause_button(self.media_player.playbackState())

    def on_slider_pressed(self):
        """Handle when user starts dragging the slider."""
        # Pause video while seeking for better control
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.was_playing = True
        else:
            self.was_playing = False

    def on_slider_released(self):
        """Handle when user releases the slider."""
        # Resume playing if it was playing before
        if self.was_playing:
            self.media_player.play()
            self.was_playing = False

    def show_image_preview(self, file_path):
        """Show image preview."""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Get the available size for the preview
                available_size = self.preview_container.size()
                if available_size.width() <= 0 or available_size.height() <= 0:
                    available_size = QSize(400, 300)  # Fallback size

                scaled_pixmap = pixmap.scaled(
                    available_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
                self.preview_label.setMinimumSize(1, 1)  # Allow label to shrink if needed

                # Show image preview and hide video widget
                self.preview_label.show()
                self.video_widget.hide()
                self.play_pause_btn.hide()
                self.stop_btn.hide()
                self.position_slider.hide()
            else:
                self.preview_label.setText('Failed to load image')
        except Exception as e:
            self.preview_label.setText(f'Error loading image: {str(e)}')

    def show_preview(self, filename):
        """Show preview for the selected file."""
        if not filename or not self.directory:
            self.clear_preview()
            return

        file_path = os.path.join(self.directory, filename)

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                self.show_video_preview(file_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
                self.show_image_preview(file_path)
            else:
                self.preview_label.setText(f'Preview not available for {ext.upper()} files')
                self.play_pause_btn.hide()
                self.stop_btn.hide()
                self.position_slider.hide()
        except Exception as e:
            self.preview_label.setText(f'Error loading preview: {str(e)}')

    def load_models(self):
        models = []

        # Get Ollama models
        ollama_models = []
        try:
            resp = requests.get('http://localhost:11434/api/tags', timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if 'models' in data:
                    ollama_models = [m['name'] for m in data['models']]
        except Exception as e:
            print(f"Warning: Could not fetch Ollama models: {e}")

        # Get LM Studio models
        lmstudio_models = []
        try:
            resp = requests.get('http://localhost:1234/v1/models', timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data:
                    seen = set()
                    for model in data['data']:
                        if 'id' in model and model['id'] not in seen:
                            seen.add(model['id'])
                            lmstudio_models.append(f"lmstudio::{model['id']}")
        except Exception as e:
            print(f"Warning: Could not fetch LM Studio models: {e}")

        models = ollama_models + lmstudio_models

        if not models:
            models = ['llava:latest']

        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.models = models
        

    def save_last_model(self):
        model = self.model_combo.currentText()
        self.settings['last_model'] = model
        self.save_settings()
        self.last_selected_model = model

    def save_last_case(self):
        idx = self.case_group.checkedId()
        case_value = self.cases[idx][1]
        self.settings['last_case'] = case_value
        self.save_settings()
        self.last_case = case_value

    def save_system_prompt(self):
        self.system_prompt = self.system_input.text()
        self.settings['system_prompt'] = self.system_prompt
        self.save_settings()

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                # Update settings dictionary with current values
                self.settings.update({
                    'last_model': self.last_selected_model,
                    'last_folder': self.last_opened_folder,
                    'rename_history': self.rename_history,
                    'last_case': self.last_case,
                    'system_prompt': self.system_prompt
                })
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def rename_files(self):
        if not self.files:
            QMessageBox.warning(self, 'No Files', 'No files to rename in the selected folder.')
            return
# ... (rest of the code remains the same)

        model = self.model_combo.currentText()
        if not model:
            QMessageBox.warning(self, 'No Model', 'Please select an AI model.')
            return

        # Check if in filename mode and custom instructions are provided
        use_filename_mode = not self.vision_mode_btn.isChecked()
        if use_filename_mode:
            custom_instructions = self.custom_instructions_input.text().strip()
            if not custom_instructions:
                QMessageBox.warning(self, 'Missing Instructions', 'Please provide custom instructions for filename-based renaming.')
                return

        case_func = {
            'kebab-case': self.to_kebab_case,
            'snake_case': self.to_snake_case,
            'camelCase': self.to_camel_case,
            'PascalCase': self.to_pascal_case,
            'Title Case': self.to_title_case,
            'lower case': self.to_lower_case,
            'UPPER CASE': self.to_upper_case
        }[self.last_case]

        self.start_file_processing(model, case_func)

    def start_file_processing(self, model, case_func):
        """Start file processing in a background thread."""
        try:
            # Stop any playing video to release file locks
            self.stop_video_for_rename()

            self.is_processing = True
            self.refresh_file_list()

            self.rename_btn.setEnabled(False)
            self.revert_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(self.files))
            self.progress_bar.setValue(0)

            current_files = self.files.copy()
            self.original_files_order = [(i, filename) for i, filename in enumerate(current_files)]

            # Get naming options
            prefix = self.prefix_input.text().strip()
            suffix = self.suffix_input.text().strip()
            add_date = self.date_checkbox.isChecked()

            current_history = {}
            use_filename_mode = not self.vision_mode_btn.isChecked()
            custom_instructions = self.custom_instructions_input.text().strip()

            self.worker = FileProcessingWorker(
                current_files, self.directory, model, case_func,
                self.system_input.text().strip(), current_history,
                prefix, suffix, add_date, use_filename_mode, custom_instructions
            )

            self.worker_thread = QThread()
            self.worker.moveToThread(self.worker_thread)

            self.worker.progress_updated.connect(self.on_progress_updated)
            self.worker.file_processed.connect(self.on_file_processed)
            self.worker.processing_finished.connect(self.on_processing_finished)
            self.worker.processing_stopped.connect(self.on_processing_stopped)
            self.worker.error_occurred.connect(self.on_error_occurred)

            self.worker_thread.started.connect(self.worker.process_files)
            self.worker_thread.start()

        except Exception as e:
            self.cleanup_processing()
            QMessageBox.critical(self, 'Error', f'Failed to start file processing: {e}')

    def stop_processing(self):
        """Stop the current file processing operation."""
        if not self.is_processing:
            return

        self.is_processing = False
        self.worker.is_running = False
        self.progress_bar.setVisible(False)
        self.stop_button.setEnabled(False)

        try:
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                if not self.worker_thread.wait(2000):
                    pass
        except Exception:
            pass

    @pyqtSlot(int, int)
    def on_progress_updated(self, current, total):
        """Handle progress updates from worker thread."""
        self.progress_bar.setValue(current + 1)

    @pyqtSlot(int, str)
    def on_file_processed(self, file_index, new_name):
        """Handle individual file processing completion."""
        if file_index < len(self.files):
            original_name = self.files[file_index]
            if not hasattr(self, 'current_rename_history'):
                self.current_rename_history = {}
            self.current_rename_history[original_name] = new_name
        self.update_file_item_realtime(file_index, new_name)

    @pyqtSlot(int)
    def on_processing_finished(self, total_processed):
        """Handle processing completion."""
        self.cleanup_processing()
        self.rename_history.update(self.current_rename_history)
        self.settings.setValue('rename_history', self.rename_history)
        self.revert_button.setEnabled(True)
        QMessageBox.information(self, 'Success', f'Renamed {total_processed} files using AI models!')

    @pyqtSlot(int)
    def on_processing_stopped(self, total_processed):
        """Handle processing being stopped early."""
        self.cleanup_processing()
        self.rename_history.update(self.current_rename_history)
        self.settings.setValue('rename_history', self.rename_history)
        self.revert_button.setEnabled(True)
        QMessageBox.information(self, 'Stopped', f'Processing stopped after renaming {total_processed} files.')

    @pyqtSlot(str, str)
    def on_error_occurred(self, filename, error_message):
        """Handle errors during processing."""
        model_type = "Unknown"
        if self.worker and hasattr(self.worker, 'model'):
            model_type = "LM Studio" if self.worker.model.startswith("lmstudio::") else "Ollama"
        QMessageBox.warning(self, 'AI Error', f'Failed to rename {filename} using {model_type}: {error_message}')

    def cleanup_processing(self):
        """Clean up after processing finishes or is stopped."""
        try:
            self.is_processing = False
            self.rename_btn.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_bar.setVisible(False)

            if self.worker_thread:
                if self.worker_thread.isRunning():
                    self.worker_thread.quit()
                    if not self.worker_thread.wait(1000):
                        pass
                self.worker_thread = None
            self.worker = None
        except Exception:
            self.is_processing = False
            self.rename_btn.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.worker_thread = None
            self.worker = None

    def revert_files(self):
        if not self.rename_history:
            QMessageBox.information(self, 'No History', 'No rename history available to revert.')
            return

        try:
            reverted_count = 0
            for original_name, new_name in self.rename_history.items():
                old_path = os.path.join(self.directory, new_name)
                new_path = os.path.join(self.directory, original_name)

                if os.path.exists(old_path):
                    counter = 1
                    while os.path.exists(new_path) and new_path != old_path:
                        base, ext = os.path.splitext(original_name)
                        new_name_with_counter = f"{base}_{counter}{ext}"
                        new_path = os.path.join(self.directory, new_name_with_counter)
                        counter += 1

                    os.rename(old_path, new_path)
                    reverted_count += 1

            self.rename_history = {}
            self.settings.setValue('rename_history', {})
            self.revert_button.setEnabled(False)

            QMessageBox.information(self, 'Success', f'Successfully reverted {reverted_count} file renames!')
            self.refresh_file_list()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to revert file renames: {e}')

    def to_kebab_case(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '-', text)
        text = re.sub(r'-+', '-', text)
        return text.strip('-')

    def to_snake_case(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = re.sub(r'_+', '_', text)
        return text.strip('_')

    def to_camel_case(self, text):
        import re
        words = re.split(r'[^a-zA-Z0-9]', text)
        words = [w for w in words if w]
        if not words:
            return text.lower()
        return words[0].lower() + ''.join(w.title() for w in words[1:])

    def to_pascal_case(self, text):
        import re
        words = re.split(r'[^a-zA-Z0-9]', text)
        words = [w for w in words if w]
        return ''.join(w.title() for w in words)

    def to_title_case(self, text):
        import re
        words = re.split(r'[^a-zA-Z0-9]', text)
        words = [w for w in words if w]
        return ''.join(w.title() for w in words)

    def to_lower_case(self, text):
        import re
        text = re.sub(r'[^a-zA-Z0-9]+', '', text)
        return text.lower()

    def to_upper_case(self, text):
        import re
        text = re.sub(r'[^a-zA-Z0-9]+', '', text)
        return text.upper()

def main():
    try:
        print("DEBUG: Starting application...")
        app = QApplication(sys.argv)
        print("DEBUG: QApplication created...")

        print("DEBUG: Creating main window...")
        window = FileRenamer()
        print("DEBUG: Main window created...")

        print("DEBUG: Showing window and starting event loop...")
        window.show()
        print("DEBUG: Window shown, executing app...")
        result = app.exec()
        print(f"DEBUG: App exited with code {result}")
        sys.exit(result)
    except Exception as e:
        print(f"ERROR: Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
