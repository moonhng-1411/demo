<#
.SYNOPSIS
    Kiểm tra xem 1 frame có phải "hub" trong CLIP visual embedding space
    không -- tức similarity trung bình cao bất thường với hầu hết mọi
    vector khác trong index, khiến nó tự nhiên lọt top-k của nhiều query
    bất kể nội dung thực sự liên quan hay không.

.PARAMETER DataDir
    Thư mục chứa faiss_clip.index, id_map.json (hoặc clip_id_map.npy),
    aic.sqlite. Mặc định .\data (theo docker-compose.yml của repo).

.PARAMETER VideoId
    video_id của frame cần kiểm tra.

.PARAMETER FrameIdx
    frame_idx của frame (field "Frame idx" trên UI).

.EXAMPLE
    .\check_hubness.ps1 -DataDir ".\data" -VideoId "L21_V014" -FrameIdx 21063
#>

param(
    [string]$IndexDir = ".\data\index",
    [string]$DbPath = ".\data\aic.sqlite",
    [Parameter(Mandatory = $true)][string]$VideoId,
    [Parameter(Mandatory = $true)][int]$FrameIdx,
    [string]$PythonExe = "python",
    [int]$SampleSize = 3000,
    [int]$BaselineFrames = 30
)

if (-not (Test-Path $IndexDir)) {
    Write-Error "Không tìm thấy thư mục index tại '$IndexDir'. Truyền đúng bằng -IndexDir."
    exit 1
}
if (-not (Test-Path $DbPath)) {
    Write-Error "Không tìm thấy aic.sqlite tại '$DbPath'. Truyền đúng bằng -DbPath."
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "check_hubness.py"
if (-not (Test-Path $scriptPath)) {
    Write-Error "Không tìm thấy check_hubness.py cùng thư mục với script này ($PSScriptRoot). Đặt 2 file cạnh nhau."
    exit 1
}

& $PythonExe $scriptPath `
    --index-dir (Resolve-Path $IndexDir).Path `
    --db-path (Resolve-Path $DbPath).Path `
    --video-id $VideoId `
    --frame-idx $FrameIdx `
    --sample-size $SampleSize `
    --baseline-frames $BaselineFrames