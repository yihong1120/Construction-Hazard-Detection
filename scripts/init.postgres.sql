-- =============================================
-- Construction-Hazard-Detection PostgreSQL baseline through 20260827.
-- Compatible with PostgreSQL 15+
-- violations.site -> FK to sites(name)
-- =============================================

SET TIME ZONE 'UTC';

DROP TABLE IF EXISTS group_features CASCADE;
DROP TABLE IF EXISTS deployment_enrollment_code_audit_logs CASCADE;
DROP TABLE IF EXISTS deployment_enrollment_codes CASCADE;
DROP TABLE IF EXISTS user_consents CASCADE;
DROP TABLE IF EXISTS email_verification_tokens CASCADE;
DROP TABLE IF EXISTS fcm_device_tokens CASCADE;
DROP TABLE IF EXISTS notifications CASCADE;
DROP TABLE IF EXISTS site_notification_preferences CASCADE;
DROP TABLE IF EXISTS violation_review_audit_logs CASCADE;
DROP TABLE IF EXISTS violation_feedback CASCADE;
DROP TABLE IF EXISTS site_media_cleanup_jobs CASCADE;
DROP TABLE IF EXISTS user_sites CASCADE;
DROP TABLE IF EXISTS site_groups CASCADE;
DROP TABLE IF EXISTS stream_configs CASCADE;
DROP TABLE IF EXISTS violations CASCADE;
DROP TABLE IF EXISTS user_identities CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS sites CASCADE;
DROP TABLE IF EXISTS legal_documents CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS group_info CASCADE;
DROP TABLE IF EXISTS deployments CASCADE;
DROP TABLE IF EXISTS tenants CASCADE;

CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(160) NOT NULL UNIQUE,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_tenants_status CHECK (status IN ('active', 'disabled'))
);

-- The seed is a bootstrap tenant only. Production tenant/deployment values
-- must be managed through the privileged deployment API and signed Registry.
-- The value is the public API root; service routes are appended by the client.
INSERT INTO tenants (id, name, description, status)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Default tenant',
    'Bootstrap tenant for local and first deployment setup.',
    'active'
);

CREATE TABLE deployments (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    api_base_url VARCHAR(2048) NOT NULL UNIQUE,
    config_revision INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_deployments_tenant
        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT,
    CONSTRAINT chk_deployments_status CHECK (status IN ('active', 'revoked')),
    CONSTRAINT chk_deployments_config_revision CHECK (config_revision >= 1)
);
CREATE INDEX idx_deployments_tenant ON deployments (tenant_id);

INSERT INTO deployments (id, tenant_id, api_base_url, config_revision, status)
VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    'https://changdar-server.mooo.com/hazard/api',
    1,
    'active'
);

CREATE TABLE deployment_enrollment_codes (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    public_id UUID NOT NULL,
    deployment_id UUID NOT NULL,
    code_verifier_hash VARCHAR(64) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    redeemed_at TIMESTAMP WITH TIME ZONE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(160) NOT NULL,
    CONSTRAINT fk_deployment_enrollment_codes_deployment
        FOREIGN KEY (deployment_id) REFERENCES deployments(id)
        ON DELETE RESTRICT,
    CONSTRAINT uq_deployment_enrollment_codes_verifier
        UNIQUE (code_verifier_hash),
    CONSTRAINT chk_deployment_enrollment_codes_expiry
        CHECK (expires_at > created_at)
);
CREATE UNIQUE INDEX uq_deployment_enrollment_codes_public_id
    ON deployment_enrollment_codes (public_id);
CREATE INDEX idx_deployment_enrollment_codes_deployment
    ON deployment_enrollment_codes (deployment_id);
CREATE INDEX idx_deployment_enrollment_codes_expiry
    ON deployment_enrollment_codes (expires_at);

CREATE TABLE deployment_enrollment_code_audit_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    enrollment_code_id BIGINT NOT NULL,
    deployment_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    actor_user_id INTEGER,
    action VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_deployment_enrollment_code_audit_code
        FOREIGN KEY (enrollment_code_id)
        REFERENCES deployment_enrollment_codes(id) ON DELETE RESTRICT,
    CONSTRAINT fk_deployment_enrollment_code_audit_deployment
        FOREIGN KEY (deployment_id) REFERENCES deployments(id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_deployment_enrollment_code_audit_tenant
        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT,
    CONSTRAINT chk_deployment_enrollment_code_audit_action
        CHECK (action IN ('created', 'revoked'))
);
CREATE INDEX idx_deployment_enrollment_code_audit_code
    ON deployment_enrollment_code_audit_logs (enrollment_code_id, created_at DESC);
CREATE INDEX idx_deployment_enrollment_code_audit_deployment
    ON deployment_enrollment_code_audit_logs (deployment_id, created_at DESC);

CREATE TABLE group_info (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    uniform_number VARCHAR(8) UNIQUE NOT NULL,
    max_allowed_streams INTEGER NOT NULL DEFAULT 8,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON COLUMN group_info.uniform_number IS 'Unified Business Number';

CREATE TABLE features (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    feature_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    email_verified_at TIMESTAMP WITH TIME ZONE,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    group_id INTEGER,
    CONSTRAINT fk_users_group
        FOREIGN KEY (group_id) REFERENCES group_info(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_users_tenant
        FOREIGN KEY (tenant_id) REFERENCES tenants(id)
        ON DELETE RESTRICT,
    CONSTRAINT chk_users_status CHECK (
        status IN (
            'active',
            'email_unverified',
            'pending_admin_approval',
            'rejected',
            'suspended'
        )
    )
);
CREATE INDEX idx_users_group ON users (group_id);
CREATE INDEX idx_users_tenant ON users (tenant_id);

ALTER TABLE deployment_enrollment_code_audit_logs
    ADD CONSTRAINT fk_deployment_enrollment_code_audit_actor
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL;

CREATE TABLE sites (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name VARCHAR(80) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_sites_name UNIQUE (name)
);

CREATE TABLE group_features (
    group_id INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, feature_id),
    CONSTRAINT fk_gf_group
        FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE,
    CONSTRAINT fk_gf_feature
        FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
);
CREATE INDEX idx_gf_group ON group_features (group_id);
CREATE INDEX idx_gf_feature ON group_features (feature_id);

CREATE TABLE site_groups (
    site_id INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (site_id, group_id),
    CONSTRAINT fk_site_groups_site
        FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE,
    CONSTRAINT fk_site_groups_group
        FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE
);
CREATE INDEX idx_sg_group ON site_groups (group_id);

CREATE TABLE user_profiles (
    user_id INTEGER PRIMARY KEY,
    family_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50),
    given_name VARCHAR(50) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    mobile_number VARCHAR(20) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_up_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX uq_user_profiles_email_lower
    ON user_profiles (lower(email));

CREATE TABLE user_identities (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL,
    provider VARCHAR(20) NOT NULL,
    provider_user_id VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    display_name VARCHAR(255),
    raw_profile JSONB,
    raw_email_is_private BOOLEAN NOT NULL DEFAULT FALSE,
    linked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_identities_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT uq_user_identities_provider_user_id
        UNIQUE (provider, provider_user_id),
    CONSTRAINT uq_user_identities_user_provider
        UNIQUE (user_id, provider)
);
CREATE INDEX idx_user_identities_user ON user_identities (user_id);

CREATE TABLE user_sites (
    user_id INTEGER NOT NULL,
    site_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, site_id),
    CONSTRAINT fk_us_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_us_site
        FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE
);
CREATE INDEX idx_us_user ON user_sites (user_id);
CREATE INDEX idx_us_site ON user_sites (site_id);

CREATE TABLE site_notification_preferences (
    user_id INTEGER NOT NULL,
    site_id INTEGER NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, site_id),
    CONSTRAINT fk_snp_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_snp_site
        FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE
);
CREATE INDEX idx_snp_site_enabled
    ON site_notification_preferences (site_id, is_enabled);
CREATE INDEX idx_snp_user ON site_notification_preferences (user_id);

CREATE TABLE notifications (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL,
    type VARCHAR(30) NOT NULL,
    title VARCHAR(120) NOT NULL,
    body TEXT NOT NULL,
    deep_link VARCHAR(255),
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_notifications_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT chk_notifications_type CHECK (
        type IN (
            'signature',
            'violation',
            'document',
            'site_alert',
            'system'
        )
    )
);
CREATE INDEX idx_notifications_user_created_id
    ON notifications (user_id, created_at DESC, id DESC);
CREATE INDEX idx_notifications_user_read_created_id
    ON notifications (user_id, is_read, created_at DESC, id DESC);
CREATE INDEX idx_notifications_user_type_created_id
    ON notifications (user_id, type, created_at DESC, id DESC);

CREATE TABLE fcm_device_tokens (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL,
    device_token_encrypted TEXT NOT NULL,
    device_token_hash VARCHAR(64) NOT NULL,
    platform VARCHAR(20) NOT NULL DEFAULT 'unknown',
    device_lang VARCHAR(20) NOT NULL,
    permission_status VARCHAR(20) NOT NULL DEFAULT 'unknown',
    app_version VARCHAR(50),
    web_vapid_key_available BOOLEAN,
    web_service_worker_registered BOOLEAN,
    last_seen_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_success_at TIMESTAMP WITH TIME ZONE,
    last_failure_at TIMESTAMP WITH TIME ZONE,
    failure_reason TEXT,
    disabled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_fcm_device_tokens_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT uq_fcm_device_tokens_token_hash
        UNIQUE (device_token_hash),
    CONSTRAINT chk_fcm_device_tokens_platform
        CHECK (platform IN ('android', 'ios', 'web', 'unknown'))
);
CREATE INDEX idx_fcm_device_tokens_user_active
    ON fcm_device_tokens (user_id, disabled_at);
CREATE INDEX idx_fcm_device_tokens_user_seen
    ON fcm_device_tokens (user_id, last_seen_at DESC);

CREATE TABLE legal_documents (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    type VARCHAR(30) NOT NULL,
    version VARCHAR(40) NOT NULL,
    locale VARCHAR(20) NOT NULL DEFAULT 'zh-TW',
    title VARCHAR(160) NOT NULL,
    content TEXT NOT NULL,
    effective_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_legal_documents_type_version_locale
        UNIQUE (type, version, locale),
    CONSTRAINT chk_legal_documents_type
        CHECK (type IN ('terms', 'privacy', 'ai_terms'))
);
CREATE INDEX idx_legal_documents_lookup
    ON legal_documents (locale, type, is_active, effective_at DESC);

CREATE TABLE user_consents (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    user_id INTEGER NOT NULL,
    terms_version VARCHAR(40) NOT NULL,
    privacy_version VARCHAR(40) NOT NULL,
    ai_terms_version VARCHAR(40) NOT NULL,
    accepted_terms BOOLEAN NOT NULL DEFAULT FALSE,
    notification_consent BOOLEAN NOT NULL DEFAULT FALSE,
    ai_terms_accepted BOOLEAN NOT NULL DEFAULT FALSE,
    accepted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ai_terms_accepted_at TIMESTAMP WITH TIME ZONE,
    notification_consent_at TIMESTAMP WITH TIME ZONE,
    ip_address VARCHAR(45),
    user_agent VARCHAR(255),
    CONSTRAINT fk_user_consents_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX idx_user_consents_user_accepted
    ON user_consents (user_id, accepted_at DESC);
CREATE INDEX idx_user_consents_ai_terms
    ON user_consents (user_id, ai_terms_accepted, accepted_at DESC);

INSERT INTO legal_documents
    (type, version, locale, title, content, effective_at, is_active)
VALUES
    (
        'terms',
        '2026-06-27',
        'zh-TW',
        '使用條款',
        '使用本服務即表示您同意遵守帳號、工地資料、影像資料與平台操作規範。您應確保提交資料合法、正確，並不得濫用服務、干擾系統或侵害他人權益。',
        '2026-06-27 00:00:00+00',
        TRUE
    ),
    (
        'privacy',
        '2026-06-27',
        'zh-TW',
        '隱私權政策',
        '我們會依服務目的處理帳號資料、工地資料、影像與偵測紀錄、通知 token、操作紀錄與同意紀錄。資料將用於安全偵測、通知、審核、維運、資安與法令遵循。',
        '2026-06-27 00:00:00+00',
        TRUE
    ),
    (
        'ai_terms',
        '2026-06-27',
        'zh-TW',
        'LLM 與 AI Agent 使用條款',
        'LLM 與 AI Agent 功能可能產生自動化分析、摘要或建議。輸出內容僅供輔助判斷，使用者仍需自行確認其正確性與適用性，並避免輸入敏感或不應揭露之資料。',
        '2026-06-27 00:00:00+00',
        TRUE
    ),
    (
        'terms',
        '2026-06-27',
        'en-US',
        'Terms of Use',
        'By using this service, you agree to follow account, site data, image data, and platform operation rules. You must submit lawful and accurate data and must not misuse the service, disrupt systems, or infringe others rights.',
        '2026-06-27 00:00:00+00',
        TRUE
    ),
    (
        'privacy',
        '2026-06-27',
        'en-US',
        'Privacy Policy',
        'We process account data, site data, images and detection records, notification tokens, activity logs, and consent records for safety detection, notifications, review, operations, security, and legal compliance.',
        '2026-06-27 00:00:00+00',
        TRUE
    ),
    (
        'ai_terms',
        '2026-06-27',
        'en-US',
        'LLM and AI Agent Terms',
        'LLM and AI Agent features may produce automated analysis, summaries, or suggestions. Outputs are for assistance only, and users remain responsible for verifying correctness and applicability while avoiding sensitive or unsuitable inputs.',
        '2026-06-27 00:00:00+00',
        TRUE
    );

UPDATE legal_documents
SET content = $terms_zh_tw$# 使用條款

版本：2026-06-27
生效日：2026-06-27

歡迎使用 Visionnaire 工地危害偵測平台。本使用條款說明您或您所代表的組織使用本平台、網站、行動應用程式、API、通知服務、影像與偵測紀錄管理功能時的權利與義務。當您建立帳號、登入、存取或使用本平台，即表示您已閱讀、理解並同意本使用條款、隱私權政策，以及註冊流程中列示的其他適用條款。

## 1. 定義

本平台指由服務提供者建置或維運，用於工地影像管理、安全危害偵測、違規紀錄、通知、審核、報表與相關輔助功能的系統。使用者指完成註冊、受邀、被指派或以其他方式取得帳號並使用本平台的人員。組織指使用者所屬或代表的公司、工地、承攬商、業主、管理單位或其他法人團體。資料包含帳號資料、工地資料、影像、影格、串流設定、偵測結果、違規紀錄、標註回饋、審核紀錄、通知 token、操作紀錄與系統日誌。

## 2. 帳號與註冊

您應提供真實、正確、完整且最新的註冊資料，並於資料變更時即時更新。您有責任妥善保管帳號、密碼、裝置與驗證資訊，並對使用您帳號進行的活動負責。若您發現帳號遭未授權使用、憑證外洩或其他安全風險，應立即通知系統管理者或服務提供者。服務提供者得基於資安、濫用防止、法令遵循或平台維運目的，暫停、限制或終止可疑帳號的存取。

## 3. 權限與組織資料

本平台可能依角色、群組、工地、站點或專案設定權限。您僅得存取您被授權查看或管理的資料，不得嘗試繞過權限、存取其他組織或其他使用者的資料。若您代表組織使用本平台，您聲明您具有代表該組織接受本條款及提交相關資料的必要權限。

## 4. 工地影像與安全偵測

本平台提供影像串流、物件偵測、違規紀錄、通知與報表等功能，用於輔助工地安全管理。偵測結果可能受鏡頭角度、光線、遮蔽、網路品質、模型版本、標註品質、環境條件或其他因素影響，因此可能產生誤判、漏判、類別錯誤或框選不準確。本平台不取代法定職安衛義務、現場安全管理、專業判斷、人工巡檢或緊急應變程序。

## 5. 通知與警示

您同意本平台得基於工地安全、違規事件、審核狀態、帳號安全、系統維運或重要服務訊息發送 App、Web、Email 或其他形式通知。安全警示與審核通知可能因裝置設定、作業系統限制、網路狀況、第三方推播服務或使用者未授權通知而延遲、失敗或無法送達。使用者與組織仍應建立必要的現場安全流程，不得單純依賴推播通知作為唯一安全措施。

## 6. 使用者回饋與標註

使用者可對偵測結果提供誤判、漏判、類別錯誤、框選錯誤或其他回饋。您提交的回饋應盡可能真實、正確且與任務相關。服務提供者得保存、審核、整理、統計及使用經授權或審核通過的回饋，用於改善產品、模型品質、資料管線與安全分析。使用者回饋不會因提交即自動成為訓練資料，是否採納得由服務提供者、組織管理者或審核流程判定。

## 7. 可接受使用

您同意不得使用本平台從事下列行為：提交不實、侵權、違法、惡意或誤導資料；上傳或散布惡意程式；干擾、掃描、攻擊、逆向工程或破壞本平台；繞過權限或安全機制；未經授權擷取、下載、轉售、揭露或分享資料；冒用他人身分；將本平台用於違反法令、契約、工地規範或公共安全的目的。

## 8. 資料授權與內容責任

您或您的組織仍保有依法屬於您或組織的資料權利。為提供、維護、保護及改善本平台，您授權服務提供者在必要範圍內儲存、處理、傳輸、備份、分析及顯示您提交或系統產生的資料。您應確保您有權提供相關影像、工地資料、人員資料與其他內容，且該等資料的提供與使用不侵害第三人權利或違反適用法令。

## 9. 個人資料與隱私

本平台對個人資料的蒐集、處理與利用依隱私權政策辦理。您理解本平台可能處理帳號資料、聯絡資料、裝置資訊、通知 token、操作紀錄、IP 位址、使用者代理字串、工地影像、偵測紀錄與審核紀錄。若您代表組織提供含個人資料的影像或紀錄，您應確保已具備適當告知、授權或其他合法基礎。

## 10. 第三方服務

本平台可能整合雲端服務、推播服務、地圖、影像儲存、身分驗證、AI 或其他第三方服務。第三方服務可能受其自身條款、限制與服務可用性影響。因第三方服務中斷、變更、限制或故障造成的延遲、資料處理失敗或功能不可用，服務提供者將盡合理努力處理，但不保證第三方服務持續無中斷。

## 11. 智慧財產權

本平台的軟體、介面、流程、文件、模型整合、系統設計、商標、標誌及其他素材，除另有約定或依法屬於使用者或組織的資料外，均由服務提供者或其授權人保有權利。未經事前書面同意，您不得複製、修改、散布、出租、出售、反向工程、移除權利聲明或以其他方式利用本平台之智慧財產。

## 12. 服務變更與可用性

服務提供者得依營運、資安、法令、技術或產品規劃需求，調整、更新、暫停或終止本平台部分或全部功能。服務提供者將盡合理努力維持系統可用性，但不保證服務永不中斷、無錯誤、完全安全或符合所有特定用途。系統維護、事故處理、網路問題、第三方服務問題或不可抗力可能造成服務暫時不可用。

## 13. 責任限制

在法律允許的最大範圍內，服務提供者不就因使用或無法使用本平台、偵測錯誤、通知延遲或未送達、資料遺失、第三方服務、現場安全事件或使用者違反本條款所生的間接、附隨、特殊、懲罰性或衍生性損害負責。本平台提供的偵測、警示、摘要、統計與建議均為輔助資訊，最終安全決策與現場管理責任仍由使用者與其組織承擔。

## 14. 賠償與合作

若因您或您的組織違反本條款、違反法令、侵犯第三人權利、提交不當資料、濫用服務或未取得必要授權，導致服務提供者、其人員、合作夥伴或其他使用者遭受請求、損害、罰鍰、費用或支出，您同意在合理範圍內協助處理並承擔相應責任。

## 15. 停權與終止

若您違反本條款、危害系統安全、侵犯他人權益、違反法令或造成平台營運風險，服務提供者得暫停、限制或終止您的帳號或部分功能。帳號停用或終止後，服務提供者仍得依法律、契約、稽核、資安、爭議處理或資料保存政策，保留必要紀錄。

## 16. 條款更新

服務提供者得不定期更新本使用條款。更新後的條款將以新版本發布於本平台或由 API 提供。若更新涉及重大權利義務變更，平台可能要求您重新確認同意。您於更新生效後繼續使用本平台，即表示您同意更新後的條款。

## 17. 準據法與管轄

本使用條款以中華民國法律為準據法。因本條款或本平台使用所生之爭議，雙方應先本於誠信協商解決；若需訴訟，除法律另有強制規定外，以臺灣臺北地方法院為第一審管轄法院。

## 18. 聯絡方式

若您對本使用條款、帳號權限、資料處理或平台使用有任何疑問，請透過平台提供的客服、系統管理者或正式聯絡管道與服務提供者聯繫。
$terms_zh_tw$
WHERE type = 'terms'
  AND locale = 'zh-TW'
  AND version = '2026-06-27';

CREATE TABLE stream_configs (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    group_id INTEGER NOT NULL,
    site_id INTEGER NOT NULL,
    stream_name VARCHAR(80) NOT NULL,
    video_url VARCHAR(255) NOT NULL,
    model_key VARCHAR(80) NOT NULL,
    detect_no_safety_vest_or_helmet BOOLEAN DEFAULT FALSE,
    detect_near_machinery_or_vehicle BOOLEAN DEFAULT FALSE,
    detect_in_restricted_area BOOLEAN DEFAULT FALSE,
    detect_in_utility_pole_restricted_area BOOLEAN DEFAULT FALSE,
    detect_machinery_close_to_pole BOOLEAN DEFAULT FALSE,
    recognition_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    expire_date TIMESTAMP WITH TIME ZONE,
    work_start_hour INTEGER,
    work_end_hour INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sc_group
        FOREIGN KEY (group_id) REFERENCES group_info(id) ON DELETE CASCADE,
    CONSTRAINT fk_sc_site
        FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE,
    CONSTRAINT uq_sc_site_stream UNIQUE (site_id, stream_name)
);
CREATE INDEX idx_sc_group ON stream_configs (group_id);
CREATE INDEX idx_sc_site ON stream_configs (site_id);

CREATE TABLE violations (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    stream_name VARCHAR(80) NOT NULL,
    detection_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255) NOT NULL,
    detections_json TEXT,
    cone_polygon_json TEXT,
    pole_polygon_json TEXT,
    warnings_json TEXT,
    stream_config_id INTEGER,
    violation_type_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    flag_reason VARCHAR(120),
    flagged_by INTEGER,
    flagged_at TIMESTAMP WITH TIME ZONE,
    review_status VARCHAR(20),
    review_note TEXT,
    reviewed_by INTEGER,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    site VARCHAR(80) NOT NULL,
    CONSTRAINT fk_vio_site_name
        FOREIGN KEY (site) REFERENCES sites(name) ON DELETE CASCADE,
    CONSTRAINT fk_vio_stream_config
        FOREIGN KEY (stream_config_id) REFERENCES stream_configs(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_vio_flagged_by
        FOREIGN KEY (flagged_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_vio_reviewed_by
        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT chk_vio_review_status CHECK (
        review_status IN ('pending', 'resolved', 'dismissed')
    )
);
CREATE INDEX idx_vio_time ON violations (detection_time);
CREATE INDEX idx_vio_site_detection_id
    ON violations (site, detection_time DESC, id DESC);
CREATE INDEX idx_vio_stream_time ON violations (stream_name, detection_time);
CREATE INDEX idx_vio_stream_config_detection_id
    ON violations (stream_config_id, detection_time DESC, id DESC);
CREATE INDEX idx_vio_type_codes ON violations USING GIN (violation_type_codes);
CREATE INDEX idx_vio_warnings_time ON violations (warnings_json, detection_time);
CREATE INDEX idx_vio_flagged_status
    ON violations (is_flagged, review_status);
CREATE INDEX idx_vio_reviewed_at ON violations (reviewed_at);

CREATE TABLE site_media_cleanup_jobs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path VARCHAR(1024) NOT NULL UNIQUE,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    lease_token VARCHAR(36),
    lease_expires_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX idx_site_media_cleanup_jobs_pending
    ON site_media_cleanup_jobs (completed_at, lease_expires_at, id)
    WHERE completed_at IS NULL;

CREATE TABLE violation_feedback (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    violation_id INTEGER NOT NULL,
    user_id INTEGER,
    anonymous_id VARCHAR(80),
    target_detection_id VARCHAR(120),
    feedback_type VARCHAR(30) NOT NULL,
    original_label VARCHAR(120),
    corrected_label VARCHAR(120),
    original_bbox JSONB,
    corrected_bbox JSONB,
    model_version VARCHAR(120),
    confidence DOUBLE PRECISION,
    note TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    reviewer_id INTEGER,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_vf_violation
        FOREIGN KEY (violation_id) REFERENCES violations(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_vf_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_vf_reviewer
        FOREIGN KEY (reviewer_id) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT chk_vf_feedback_type CHECK (
        feedback_type IN (
            'false_positive',
            'false_negative',
            'wrong_class',
            'bad_bbox'
        )
    ),
    CONSTRAINT chk_vf_status CHECK (
        status IN ('pending', 'reviewed', 'accepted', 'rejected')
    )
);
CREATE INDEX idx_vf_violation_created
    ON violation_feedback (violation_id, created_at);
CREATE INDEX idx_vf_type_status
    ON violation_feedback (feedback_type, status);
CREATE INDEX idx_vf_user_created
    ON violation_feedback (user_id, created_at);
CREATE INDEX idx_vf_status_created
    ON violation_feedback (status, created_at);

CREATE TABLE violation_review_audit_logs (
    id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    violation_id INTEGER NOT NULL,
    action VARCHAR(40) NOT NULL DEFAULT 'review_status_changed',
    old_status VARCHAR(20),
    new_status VARCHAR(20) NOT NULL,
    review_note TEXT,
    flagged_reason VARCHAR(120),
    reviewed_by INTEGER,
    reviewed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_vral_violation
        FOREIGN KEY (violation_id) REFERENCES violations(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_vral_reviewer
        FOREIGN KEY (reviewed_by) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT chk_vral_new_status CHECK (
        new_status IN ('pending', 'resolved', 'dismissed')
    )
);
CREATE INDEX idx_vral_violation_time
    ON violation_review_audit_logs (violation_id, reviewed_at);
CREATE INDEX idx_vral_action_time
    ON violation_review_audit_logs (action, reviewed_at);
CREATE INDEX idx_vral_reviewer_time
    ON violation_review_audit_logs (reviewed_by, reviewed_at);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_group_info_updated_at
BEFORE UPDATE ON group_info
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_features_updated_at
BEFORE UPDATE ON features
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_tenants_updated_at
BEFORE UPDATE ON tenants
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_deployments_updated_at
BEFORE UPDATE ON deployments
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE FUNCTION increment_deployment_config_revision()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
       OR NEW.api_base_url IS DISTINCT FROM OLD.api_base_url
       OR NEW.status IS DISTINCT FROM OLD.status THEN
        NEW.config_revision = OLD.config_revision + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_deployments_config_revision
BEFORE UPDATE ON deployments
FOR EACH ROW EXECUTE FUNCTION increment_deployment_config_revision();

CREATE TRIGGER trg_sites_updated_at
BEFORE UPDATE ON sites
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_user_profiles_updated_at
BEFORE UPDATE ON user_profiles
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_site_notification_preferences_updated_at
BEFORE UPDATE ON site_notification_preferences
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_stream_configs_updated_at
BEFORE UPDATE ON stream_configs
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE FUNCTION notify_stream_config_changed()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('stream_config_changed', OLD.id::TEXT);
        RETURN OLD;
    END IF;

    PERFORM pg_notify('stream_config_changed', NEW.id::TEXT);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_stream_configs_changed
AFTER INSERT OR UPDATE OR DELETE ON stream_configs
FOR EACH ROW EXECUTE FUNCTION notify_stream_config_changed();

CREATE TRIGGER trg_fcm_device_tokens_updated_at
BEFORE UPDATE ON fcm_device_tokens
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO group_info (id, name, uniform_number, max_allowed_streams)
VALUES (1, 'Default Group', '00000001', 8)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    max_allowed_streams = EXCLUDED.max_allowed_streams;

SELECT setval(
    pg_get_serial_sequence('group_info', 'id'),
    GREATEST((SELECT MAX(id) FROM group_info), 1)
);

INSERT INTO features (feature_name, description)
VALUES ('yolo_api', 'Utilising YOLO for real-time object detection.')
ON CONFLICT (feature_name) DO UPDATE SET
    description = EXCLUDED.description;

INSERT INTO users (username, password_hash, role, status, group_id, tenant_id)
VALUES (
    'user',
    '$argon2id$v=19$m=65536,t=3,p=4$WWrgNzRESjrJxeP6KC+jsQ$LRWIP3bk3vAJf5kSEA+gkSk1+KYvVU2VDwCKGiUtBCg',
    'admin',
    'active',
    1,
    '00000000-0000-0000-0000-000000000001'
)
ON CONFLICT (username) DO UPDATE SET
    password_hash = EXCLUDED.password_hash,
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    group_id = EXCLUDED.group_id,
    tenant_id = EXCLUDED.tenant_id;

INSERT INTO group_features (group_id, feature_id)
SELECT 1, f.id
FROM features f
WHERE f.feature_name = 'yolo_api'
ON CONFLICT (group_id, feature_id) DO NOTHING;
