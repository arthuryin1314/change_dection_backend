$ErrorActionPreference = 'Stop'
$ghExe = 'C:\Users\Administrator\AppData\Local\Temp\codex-gh-land-cover\bin\gh.exe'
$repo = 'arthuryin1314/change_dection_backend'
$files = @{8='01b-area-display.md';9='02-matrix.md';11='04-auto-identification.md';12='05-history.md'}
function Replace-Required([string]$text, [string]$old, [string]$new) {
    if (-not $text.Contains($old)) { throw ('Expected text missing: ' + $old) }
    return $text.Replace($old,$new)
}
foreach ($n in @(6,8,9,11,12)) {
    $issue = Get-Content -LiteralPath (Join-Path $env:TEMP "land-cover-current-$n.json") -Raw | ConvertFrom-Json
    $body = $issue.body
    switch ($n) {
        6 {
            $body = Replace-Required $body '扩展单期地物识别，在保留分类色块展示的同时，计算并保存整幅影像有效区域内的六类面积及空间分类结果。' '地物识别页面保持识别与分类色块展示，不新增六类面积面板。变化检测页面分为“变化前”和“变化后”两个区域，两侧各自选择影像，在两期共用的所选模型下自动关联匹配的已有识别结果，展示各自整幅有效区域的六类面积；用户不直接选择识别结果记录。'
            $body = Replace-Required $body '- 地物识别页面展示六类面积及单位选择，保留分类显示控制。' '- 六类面积展示在变化检测页面的前后两个区域；地物识别主页面不新增面积面板。两侧各自选择影像，沿用两期共用一个模型；布局细节不固定。'
            $body += @'

## 2026-09-08 页面与结果关联修订

- 选择影像仅按完整身份查询已有识别结果，不触发识别；分类存在但缺面积汇总时，可通过单独请求从保存栅格统计并保存面积，不重跑模型。#11 在用户发起变化检测后负责缺失分类的自动补算。
- 每侧面积标为“整幅有效区域”，矩阵为“共同有效区域”，不要求前后单期总面积等于矩阵行列合计。
- #8 拥有两侧选择状态、只读结果解析入口和唯一前端单位换算；#9 复用这些能力增加矩阵，并在执行前重新校验身份及完整性；#11 负责补算编排，#12 保留两类历史列表职责。
- 完整身份沿用 #7 的用户、影像内容版本、权重内容版本、规范化参数、分类体系版本、pipeline_version 与 grid_policy_version，不以 image_id 最新记录代替匹配。
- #9 的完整交付依赖 #8 的查询与选择状态契约；其矩阵计算核心仍只依赖 #7。不重开 #7。
'@
        }
        8 {
            $newWhat = @'
用户在变化检测页面分别选择“变化前”和“变化后”影像，系统在两期共用的所选模型下按完整身份自动关联各自已有识别结果，显示该结果的六类面积。每侧统计整幅有效区域，默认公顷，可切换平方米和平方公里；用户不直接选择结果记录。地物识别页面保留识别与色块展示，不新增面积面板。已有分类仅缺面积汇总时，通过独立统计请求从保存栅格计算并持久化平方米，不重跑推理。交付后续矩阵复用的选择／结果状态、只读解析入口及唯一前端单位转换。
'@
            $body = [regex]::Replace($body,'(?s)(## What to build\s*).*?(\s*## Acceptance criteria)',{param($m) $m.Groups[1].Value + $newWhat + $m.Groups[2].Value})
            $body += @'

## 页面与只读结果解析验收（2026-09-08）

- [ ] 前后两侧分别选择影像，共用一个模型；只展示完整身份匹配结果的六类面积，并标明“整幅有效区域”。未匹配明确显示未就绪，不显示其他影像或旧模型面积。
- [ ] 切换影像立即清除该侧旧数据，切换模型使两侧失效。通过请求序号及身份比对忽略迟到响应；渲染与面积固定到相同 resultId 和 identitySha256。
- [ ] 本任务拥有只读解析入口，复用 #7 的 get_by_identity 和完整身份构造，包括 pipeline_version、grid_policy_version。命中后读取详情并只读验证文件，不能按 image_id 取最新结果。
- [ ] 只读解析不得写库、认领、创建分类结果或推理。避免直接调用会 flush 哈希的源哈希解析封装、会 invalidate/commit 的详情损坏路径及结果创建接口；复用无副作用的底层哈希与文件验证能力，哈希未缓存时在内存计算。
- [ ] 查询区分匹配、缺失、处理中、失败、不可用；仅完整匹配返回可用结果及面积状态。未授权遵循现有不泄露资源存在性的错误规则。具体状态命名由实现确定，#9 映射为矩阵统一错误码。
- [ ] 若新增静态解析路由，确保不被动态结果 ID 路由遮蔽。查询缺失不生成；分类匹配但面积缺失时，单独按 result_id 统计并幂等保存，统计失败不使有效分类结果失效。
- [ ] 两侧共用单位转换能力，默认公顷；共用一个单位选择器只是布局建议。显示切换不发起推理或重新统计。
- [ ] API 与前端会话测试覆盖完整身份、哈希变化、文件不可用、无查询写入副作用、两侧选择、模型切换、迟到响应、缺失分类、仅缺面积和单位换算。保留原有分块统计、地理面积与真实性能验收。
- [ ] 不提前实现 #11 自动补算或 #12 历史列表；#8 的用户选择对象是影像，不是历史结果记录。
'@
        }
        9 {
            $body = [regex]::Replace($body,'(?m)^- \[ \] 本任务完整拥有原任务 4a.*$','- [ ] 本任务复用 #8 的完整身份只读解析能力，不重复实现版本查找；检测前重新验证当前用户、影像和模型版本及结果完整性。两期匹配完整结果时不调用推理，直接比较。原 #10 的复用验收由 #8 查询与本任务矩阵集成共同覆盖。')
            $body = [regex]::Replace($body,'(?m)^- \[ \] 行为前期、列为后期.*$','- [ ] 行为前期、列为后期，按固定六类顺序显示全部 36 格。复用 #8 的页面骨架、前后选择状态和单位转换，默认公顷并支持平方米、平方公里，不另建选择状态或换算实现。')
            $body = [regex]::Replace($body,'(?m)^- 任务 1a.*不构成硬依赖。\s*$','- #7：分类结果契约与共享像元地表面积能力。' + "`n" + '- #8：前后选择状态及完整身份只读解析入口，属于具体接口依赖；矩阵计算核心可先基于 #7 开发，但整卡交付需 #8。')
            $body += @'

## 页面职责边界（2026-09-08）

- [ ] #8 负责变化前／变化后影像选择及单期面积，#9 仅在相同页面状态上增加矩阵，禁止重复实现页面骨架和已有结果解析。
- [ ] 单期面板明确展示各期整幅有效区域面积，矩阵明确展示共同有效区域转移面积；行列合计不要求等于单期整幅分类面积。
- [ ] 点击检测时重新校验所选身份与完整性，将 #8 的只读解析状态映射到本任务统一错误码清单；缺失结果仍由 #11 补算。
'@
        }
        11 {
            $body += @'

## 前后区域接入（2026-09-08）

- [ ] 接入 #8 已有的前后影像、共用模型及结果状态；由 #8 只读解析识别未就绪情况，由本任务在用户发起检测后自动识别并保存。选择影像本身不触发本任务补算。
- [ ] 补算后将完整结果及面积状态交回同一前后区域，两期均完整就绪才执行 #9；面积缺失调用 #8 的单独统计能力，不通过重新推理获取面积。
- [ ] 不复制 #8 的查询／身份构造或前端选择状态，不把只读关联已有结果误当作已经完成自动补算。本任务经 #9 传递依赖 #8。
'@
        }
        12 {
            $body += @'

## 与面积面板的边界（2026-09-08）

- [ ] #8 在变化检测页面按两侧所选影像自动关联已有结果，用户不直接选择结果记录；该流程不替代本任务的单期识别与变化检测历史列表。
- [ ] 本任务保留已确认的历史入口与单期历史详情；复用 #8 的面积展示和转换能力，不在地物识别主页面新增常驻六类面积面板。
'@
        }
    }
    $bodyFile = Join-Path $env:TEMP "land-cover-placement-$n.md"
    Set-Content -LiteralPath $bodyFile -Value $body -Encoding utf8
    & $ghExe issue edit $n --repo $repo --body-file $bodyFile | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Update failed #$n" }
    $actual = & $ghExe issue view $n --repo $repo --json body --jq .body
    if ($LASTEXITCODE -ne 0 -or ($actual -join "`n").Trim() -ne $body.Replace("`r`n","`n").Trim()) { throw "Verify failed #$n" }
    if ($n -eq 6) { $local = Join-Path $PSScriptRoot '../land-cover-transition-spec.md' } else { $local = Join-Path $PSScriptRoot $files[$n] }
    Set-Content -LiteralPath $local -Value ("# " + $issue.title + "`n`n" + $body) -Encoding utf8
    Write-Output "Updated and verified #$n"
}
$id = & $ghExe api "repos/$repo/issues/8" --jq .id
if ($LASTEXITCODE -ne 0) { throw 'ID lookup failed' }
$deps = & $ghExe api "repos/$repo/issues/9/dependencies/blocked_by" --jq '.[].number'
if ($LASTEXITCODE -ne 0) { throw 'Dependencies lookup failed' }
if (@($deps) -notcontains '8') {
    & $ghExe api --method POST "repos/$repo/issues/9/dependencies/blocked_by" -F "issue_id=$id" --silent
    if ($LASTEXITCODE -ne 0) { throw 'Dependency add failed' }
}
& $ghExe api "repos/$repo/issues/9/dependencies/blocked_by" --jq '.[].number'
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed' }
