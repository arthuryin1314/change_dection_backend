$ErrorActionPreference = 'Stop'
$ghExe = 'C:\Users\Administrator\AppData\Local\Temp\codex-gh-land-cover\bin\gh.exe'
$repo = 'arthuryin1314/change_dection_backend'
$map = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'published.json') -Raw | ConvertFrom-Json -AsHashtable
$repoSection = @'

## Repository scope

- 后端：[arthuryin1314/change_dection_backend](https://github.com/arthuryin1314/change_dection_backend)。本 Issue 在后端仓库跟踪前后端联合交付。
- 前端：[arthuryin1314/change_dection](https://github.com/arthuryin1314/change_dection)。页面、交互与前端会话测试在此前端仓库完成。
- 涉及两端时应在两个仓库分别验证并将对应变更关联本 Issue；不能把前端代码写入后端仓库，也不能因后端单独通过而宣告跨仓验收完成。仅做后端契约的改动无需制造无意义前端修改。
'@
foreach ($key in @('1a','1b','2','4b','5','3','4a')) {
    $entry = $map[$key]
    $sourcePath = Join-Path $PSScriptRoot $entry.file
    $source = Get-Content -LiteralPath $sourcePath -Raw
    if (-not $source.Contains('## Repository scope')) {
        $source = $source.TrimEnd() + "`n" + $repoSection + "`n"
        Set-Content -LiteralPath $sourcePath -Value $source -Encoding utf8
    }
    $title = ($source -split '\r?\n',2)[0].Substring(2)
    $body = $source.Substring($source.IndexOf('## Parent'))
    if ($key -eq '4a') { $body = "本任务职责已全部合并至 #9，不再单独实施；这是任务合并，不代表功能已实现。#11 改为依赖 #9。`n`n" + $body }
    foreach ($refKey in $map.Keys) {
        $body = [regex]::Replace($body, ('任务 ' + [regex]::Escape($refKey) + '(?![a-z0-9])'), ('任务 ' + $refKey + '（#' + $map[$refKey].number + '）'))
    }
    $bodyPath = Join-Path $env:TEMP ('land-cover-revised-' + $entry.number + '.md')
    Set-Content -LiteralPath $bodyPath -Value $body -Encoding utf8
    & $ghExe issue edit $entry.number --repo $repo --title $title --body-file $bodyPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw ('Issue update failed: ' + $entry.number) }
    $map[$key].title = $title
    Write-Output ('Updated #' + $entry.number)
}
foreach ($edge in @(@{target=11; blocker=9}, @{target=12; blocker=8})) {
    $id = & $ghExe api "repos/$repo/issues/$($edge.blocker)" --jq '.id'
    if ($LASTEXITCODE -ne 0) { throw 'ID lookup failed' }
    $existing = & $ghExe api "repos/$repo/issues/$($edge.target)/dependencies/blocked_by" --jq '.[].number'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency lookup failed' }
    if (@($existing) -notcontains [string]$edge.blocker) {
        & $ghExe api --method POST "repos/$repo/issues/$($edge.target)/dependencies/blocked_by" -F "issue_id=$id" --silent
        if ($LASTEXITCODE -ne 0) { throw 'Dependency add failed' }
    }
}
$oldId = & $ghExe api "repos/$repo/issues/10" --jq '.id'
if ($LASTEXITCODE -ne 0) { throw 'Old dependency ID lookup failed' }
$existing = & $ghExe api "repos/$repo/issues/11/dependencies/blocked_by" --jq '.[].number'
if ($LASTEXITCODE -ne 0) { throw 'Old dependency lookup failed' }
if (@($existing) -contains '10') {
    & $ghExe api --method DELETE "repos/$repo/issues/11/dependencies/blocked_by/$oldId" --silent
    if ($LASTEXITCODE -ne 0) { throw 'Old dependency removal failed' }
}
& $ghExe issue edit 10 --repo $repo --remove-label ready-for-agent | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Merged label removal failed' }
& $ghExe issue close 10 --repo $repo --reason 'not planned'
if ($LASTEXITCODE -ne 0) { throw 'Merged issue close failed' }
$map['4a'].status = 'merged-into-9'
$map | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'published.json') -Encoding utf8
