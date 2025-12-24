import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import styles from './ChatInterface.module.css';
import { ChatService } from '../../services/chatService';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Array<{
    title: string;
    score: number;
    content_preview: string;
  }>;
}

const ChatInterface: React.FC = () => {
  const { siteConfig } = useDocusaurusContext();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [showSources, setShowSources] = useState<{[key: string]: boolean}>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize chat service with backend URL from site config
  const chatService = new ChatService(
    (siteConfig.customFields?.backendUrl as string) || 'http://localhost:8000'
  );

  // Load initial welcome message
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      role: 'assistant',
      content: 'Welcome to the Physical AI & Humanoid Robotics textbook chat! I can answer questions based on the textbook content. What would you like to know?',
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API using the service
      const response = await chatService.sendMessage({
        message: inputValue,
        conversation_id: conversationId || undefined,
        user_id: 'docusaurus-user', // In a real app, this would come from auth
      });

      // Update conversation ID if new one was created
      if (response.conversation_id && !conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add assistant message to chat
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp,
        sources: response.sources,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetConversation = async () => {
    try {
      const response = await chatService.resetConversation();
      setConversationId(response.conversation_id);

      // Reset messages with welcome message
      const welcomeMessage: Message = {
        id: 'welcome',
        role: 'assistant',
        content: 'Welcome! I\'ve reset our conversation. What would you like to know about Physical AI & Humanoid Robotics?',
        timestamp: new Date(),
      };

      setMessages([welcomeMessage]);
    } catch (error) {
      console.error('Error resetting conversation:', error);
    }
  };

  const toggleSources = (messageId: string) => {
    setShowSources(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <h2>Physical AI & Humanoid Robotics Chat</h2>
        <button
          className={clsx('button button--secondary', styles.resetButton)}
          onClick={handleResetConversation}
        >
          New Chat
        </button>
      </div>

      <div className={styles.messagesContainer}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              styles.message,
              styles[message.role],
              message.role === 'user' ? styles.userMessage : styles.assistantMessage
            )}
          >
            <div className={styles.messageHeader}>
              <span className={styles.roleLabel}>
                {message.role === 'user' ? 'You' : 'AI Assistant'}
              </span>
              <span className={styles.timestamp}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <div className={styles.messageContent}>
              {message.content}
            </div>

            {message.sources && message.sources.length > 0 && (
              <div className={styles.sourcesSection}>
                <button
                  className={styles.sourcesToggle}
                  onClick={() => toggleSources(message.id)}
                >
                  {showSources[message.id] ? 'Hide Sources' : 'Show Sources'} ({message.sources.length})
                </button>

                {showSources[message.id] && (
                  <div className={styles.sourcesList}>
                    <h4>Sources from Textbook:</h4>
                    {message.sources.map((source, index) => (
                      <div key={index} className={styles.sourceItem}>
                        <div className={styles.sourceTitle}>{source.title}</div>
                        <div className={styles.sourcePreview}>
                          {source.content_preview}
                        </div>
                        <div className={styles.sourceScore}>
                          Relevance: {(source.score * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className={clsx(styles.message, styles.assistantMessage)}>
            <div className={styles.messageHeader}>
              <span className={styles.roleLabel}>AI Assistant</span>
            </div>
            <div className={styles.typingIndicator}>
              <div className={styles.dot}></div>
              <div className={styles.dot}></div>
              <div className={styles.dot}></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about Physical AI & Humanoid Robotics..."
          className={styles.inputField}
          disabled={isLoading}
        />
        <button
          type="submit"
          className={clsx('button button--primary', styles.sendButton)}
          disabled={!inputValue.trim() || isLoading}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;