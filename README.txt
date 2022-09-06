Setup:
  - (Optional) setup virtual environment:
    > dari command line navigasi ke folder ini
    > run command: python -m venv .venv
    > aktivasi (windows): .\.venv\Script\activate
  - Install library yang diperlukan:
    > run command: pip install requirements.txt

Penggunaan:
  - tanpa argumen maka input videonya dari camera:
    > python main.py
  - dengan input argumen:
    > python main.py --input videoname.videoextension
  - jika ingin output ke file:
    > python main.py --output videoname.videoextension
  - jika ingin proses tanpa display:
    > python main.py --output videoname.videoextension --display 0