@echo off
rem Windows에서 기존 데스크톱 관제를 실행하는 배치 파일이다. 프로젝트 위치와 실행 환경을 준비한 뒤 Python 진입점을 호출한다.
cd /d "%~dp0"
.venv\Scripts\python.exe main.py
pause
