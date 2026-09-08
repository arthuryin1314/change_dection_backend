$ErrorActionPreference = 'Stop'
$ghExe = 'C:\Users\Administrator\AppData\Local\Temp\codex-gh-land-cover\bin\gh.exe'
$repo = 'arthuryin1314/change_dection_backend'
$ticketDefs = @(
    @{ key='1a'; file='01-single-period.md'; blockers=@() },
    @{ key='1b'; file='01b-area-display.md'; blockers=@('1a') },
    @{ key='2'; file='02-matrix.md'; blockers=@('1a','1b') },
    @{ key='4b'; file='04-auto-identification.md'; blockers=@('2','1a') },
    @{ key='5'; file='05-history.md'; blockers=@('2','1b') },
    @{ key='3'; file='03-grid-alignment.md'; blockers=@('2') }
)
$manifestPath = Join-Path $PSScriptRoot 'published.json'
$published = @{}
if (Test-Path -LiteralPath $manifestPath) {
    $published = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
}
foreach ($ticket in $ticketDefs) {
    if ($published.ContainsKey($ticket.key)) { continue }
    $source = Get-Content -LiteralPath (Join-Path $PSScriptRoot $ticket.file) -Raw
    $title = ($source -split '\r?\n',2)[0].Substring(2)
    $body = $source.Substring($source.IndexOf('## Parent'))
    foreach ($key in $published.Keys) {
        $body = [regex]::Replace($body, ('任务 ' + [regex]::Escape($key) + '(?![a-z0-9])'), ('任务 ' + $key + '（#' + $published[$key].number + '）'))
    }
    $bodyPath = Join-Path $env:TEMP ('land-cover-ticket-' + $ticket.key + '.md')
    Set-Content -LiteralPath $bodyPath -Value $body -Encoding utf8
    $url = & $ghExe issue create --repo $repo --title $title --body-file $bodyPath --label ready-for-agent
    if ($LASTEXITCODE -ne 0) { throw ('Issue create failed: ' + $ticket.key) }
    $url = ($url | Select-Object -Last 1).Trim()
    $number = [int]($url.Split('/')[-1])
    $published[$ticket.key] = @{ number=$number; url=$url; file=$ticket.file; title=$title }
    $published | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Output ($ticket.key + ': ' + $url)
}
foreach ($ticket in $ticketDefs) {
    $entry = $published[$ticket.key]
    $source = Get-Content -LiteralPath (Join-Path $PSScriptRoot $ticket.file) -Raw
    $body = $source.Substring($source.IndexOf('## Parent'))
    foreach ($key in $published.Keys) {
        $body = [regex]::Replace($body, ('任务 ' + [regex]::Escape($key) + '(?![a-z0-9])'), ('任务 ' + $key + '（#' + $published[$key].number + '）'))
    }
    $bodyPath = Join-Path $env:TEMP ('land-cover-ticket-' + $ticket.key + '.md')
    Set-Content -LiteralPath $bodyPath -Value $body -Encoding utf8
    & $ghExe issue edit $entry.number --repo $repo --body-file $bodyPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw ('Issue body update failed: ' + $ticket.key) }
    $existing = & $ghExe api ('repos/' + $repo + '/issues/' + $entry.number + '/dependencies/blocked_by') --jq '.[].number'
    if ($LASTEXITCODE -ne 0) { throw ('Dependency lookup failed: ' + $ticket.key) }
    foreach ($blocker in $ticket.blockers) {
        $blockNumber = $published[$blocker].number
        if (@($existing) -contains [string]$blockNumber) { continue }
        $issueId = & $ghExe api ('repos/' + $repo + '/issues/' + $blockNumber) --jq '.id'
        if ($LASTEXITCODE -ne 0) { throw 'Blocker ID lookup failed' }
        & $ghExe api --method POST ('repos/' + $repo + '/issues/' + $entry.number + '/dependencies/blocked_by') -F ('issue_id=' + $issueId) --silent
        if ($LASTEXITCODE -ne 0) { throw ('Native dependency failed: ' + $ticket.key) }
    }
    Write-Output ('Linked: ' + $entry.number)
}
