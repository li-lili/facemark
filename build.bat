@echo off
chcp 65001 >nul
echo ==============================================
echo 开始编译Qt资源文件...
echo ==============================================
if %errorlevel% neq 0 (
    echo 资源文件编译失败！
    pause
    exit /b 1
)

echo ==============================================
echo 开始打包可执行文件...
echo ==============================================
:: 2. PyInstaller打包（关键参数说明）
:: -F：单文件
:: -w：无控制台
:: --hidden-import=resources_rc：强制打包资源模块
:: --name：自定义可执行文件名
:: --clean：清理缓存
:: --noupx：禁用UPX压缩（避免兼容问题）
pyinstaller -F -w ^
--name "舵机调试工具" ^
--hidden-import=resources_rc ^
--clean ^
--noupx ^
D:\testDemo\test\run_interface.py

echo ==============================================
echo 清理临时文件...
echo ==============================================
:: 3. 清理打包产生的临时文件（静默删除，不存在则忽略）
if exist "build" rmdir /s /q "build" >nul 2>&1
if exist "run_interface.spec" del /f "run_interface.spec" >nul 2>&1

:: 4. 可选：如果需要清空旧的dist目录（打包前清理）
:: if exist "dist" rmdir /s /q "dist" >nul 2>&1

echo ==============================================
echo 打包完成！可执行文件位于 dist\舵机调试工具.exe
echo ==============================================
pause