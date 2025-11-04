-- Message store (LangChain compatible with Azure SQL)
CREATE TABLE message_store (
    session_id NVARCHAR(255),
    message NVARCHAR(MAX),  -- Instead of JSONB
    created_at DATETIME2 DEFAULT GETUTCDATE()
);

-- Session tracking
CREATE TABLE conversation_sessions (
    session_id NVARCHAR(50) PRIMARY KEY,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    last_updated DATETIME2 DEFAULT GETUTCDATE(),
    total_messages INT DEFAULT 0,
    user_info NTEXT
);

-- Indexes for performance
CREATE INDEX IX_message_store_session_id ON message_store(session_id);
CREATE INDEX IX_conversation_sessions_created_at ON conversation_sessions(created_at);
