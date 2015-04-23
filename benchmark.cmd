@echo off
set LOG=benchmark.log

rem Find executable.

set EXE=rt_x64_Release.exe
if not exist %EXE% set EXE=NTrace_Win32_Release.exe
if not exist %EXE% set EXE=NTrace_Win32_Debug.exe

rem Benchmark conference, fairyforest, and sibenik.

%EXE% benchmark --log=%LOG% --mesh=scenes/rt/misc/teapot.obj --sbvh-alpha=1.0e-5 --ao-radius=5 --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100"

echo Done.
PAUSE
