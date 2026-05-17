$ErrorActionPreference = "Stop"

$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$pythonExe = "D:\Anaconda\envs\serverless_rl\python.exe"
$port = 8000
$url = "http://localhost:$port/defense_demo.html"

Set-Location -LiteralPath $projectRoot

Write-Host "[1/3] 项目目录: $projectRoot"
Write-Host "[2/3] 启动本地静态服务: $pythonExe -m http.server $port"

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$projectRoot'; & '$pythonExe' -m http.server $port"
)

Start-Sleep -Seconds 2

Write-Host "[3/3] 打开演示页面: $url"
Start-Process $url

Write-Host ""
Write-Host "如果页面已打开，后续可在当前终端继续运行现场评估命令。"
Write-Host "评估完成后回到浏览器，按 Ctrl+F5 强制刷新即可。"
