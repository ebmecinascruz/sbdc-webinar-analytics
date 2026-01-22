from PyInstaller.utils.hooks import copy_metadata, collect_all

datas, binaries, hiddenimports = collect_all("streamlit")
datas += copy_metadata("streamlit")
