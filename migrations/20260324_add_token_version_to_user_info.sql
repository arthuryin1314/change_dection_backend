-- 为 JWT 失效控制添加版本号字段
ALTER TABLE user_info
ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;

-- 旧的 usertoken 表不再参与认证流程。
-- 如果确认线上已完成 JWT 切换，可再手动评估是否归档或删除该表。

