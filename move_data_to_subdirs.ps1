#Usage move all the data, which includes bonsai workflow, trial information, camera timestamp and video, into standby_folder and then run this script with -StartID and -ExperimentDate.
#for example .\move_data_to_subdirs.ps1 -StartID 26146 -ExperimentDate 260320
#Note1: transfer files from different date seperately
#Note2: avoiding transferring other files, no matter it is a failed experiment that does not store much information or the original bonsai workflow
#Note3: ensure the -StartID does not overlap with the existing ID.
param( 
    # [Parameter(Mandatory = $true)]
    # [string]$SourceFolder,

    # [Parameter(Mandatory = $true)]
    # [string]$DestinationRoot,   # <-- server location

    [Parameter(Mandatory=$true)]
    [int]$StartID,

    [Parameter(Mandatory=$true)]
    [string]$ExperimentDate
)

# ---- Config ----
$ExperimentName = "choices"
$SessionNumber  = "session1"
$SourceFolder="C:\Users\einat\Desktop\BonVision\standby_folder"
$DestinationRoot="Z:\DATA\experiment_trackball_Optomotor\Zball"
# ---- Validate paths ----
foreach ($path in @($SourceFolder, $DestinationRoot)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        Write-Error "Folder does not exist: $path"
        exit 1
    }
}

$SourceFolder    = (Resolve-Path -LiteralPath $SourceFolder).Path
$DestinationRoot = (Resolve-Path -LiteralPath $DestinationRoot).Path

# ---- Timestamp extraction ----
function Get-TimestampFolderName {
    param([string]$FileName)

    if ($FileName -match '(\d{4}-\d{2}-\d{2}T\d{2}_\d{2})_\d{2}') {
        return $matches[1]
    }

    if ($FileName -match '(\d{8})_(\d{6})') {
        $datePart = $matches[1]
        $timePart = $matches[2]

        return "{0}-{1}-{2}T{3}_{4}" -f `
            $datePart.Substring(0,4),
            $datePart.Substring(4,2),
            $datePart.Substring(6,2),
            $timePart.Substring(0,2),
            $timePart.Substring(2,2)
    }

    return $null
}

# ---- STEP 1: Group into timestamp folders INSIDE SourceFolder ----
Write-Host "Step 1: Grouping files inside source folder..."

Get-ChildItem -LiteralPath $SourceFolder -File -Recurse | ForEach-Object {

    $file = $_
    $timestampFolder = Get-TimestampFolderName -FileName $file.Name

    if ($null -ne $timestampFolder) {

        $targetFolder = Join-Path $SourceFolder $timestampFolder

        if (-not (Test-Path -LiteralPath $targetFolder)) {
            New-Item -ItemType Directory -Path $targetFolder | Out-Null
        }

        $destinationPath = Join-Path $targetFolder $file.Name

        if ($file.Directory.FullName -ne $targetFolder) {
            Move-Item -LiteralPath $file.FullName `
                      -Destination $destinationPath `
                      -Force
        }
    }
}

Write-Host "Timestamp grouping complete."

# ---- STEP 2: Move timestamp folders (batches) to server ----
Write-Host "Step 2: Moving batches to server..."

$CurrentID = $StartID

# Only timestamp folders (exclude GN folders)
$BatchFolders = Get-ChildItem -Path $SourceFolder -Directory | 
    Where-Object { $_.Name -notlike "GN*" }

foreach ($Batch in $BatchFolders) {

    $Files = Get-ChildItem -Path $Batch.FullName -File

    if ($Files.Count -gt 0) {

        $ANIMAL_FOLDER = "GN$CurrentID"

        $TargetPath = Join-Path $DestinationRoot `
            "$ANIMAL_FOLDER\$ExperimentDate\$ExperimentName\$SessionNumber"

        New-Item -Path $TargetPath -ItemType Directory -Force | Out-Null

        foreach ($File in $Files) {

            $DestinationPath = Join-Path $TargetPath $File.Name

            Move-Item -LiteralPath $File.FullName `
                      -Destination $DestinationPath `
                      -Force

            # Optional safety cleanup
            if ((Test-Path $DestinationPath) -and (Test-Path $File.FullName)) {
                Remove-Item -LiteralPath $File.FullName -Force
            }
        }

        # Remove empty batch folder
        if ((Get-ChildItem -Path $Batch.FullName -Force).Count -eq 0) {
            Remove-Item -LiteralPath $Batch.FullName -Force
        }

        Write-Host "Batch '$($Batch.Name)' -> $ANIMAL_FOLDER"
        $CurrentID++
    }
}

Write-Host "All batches processed successfully."