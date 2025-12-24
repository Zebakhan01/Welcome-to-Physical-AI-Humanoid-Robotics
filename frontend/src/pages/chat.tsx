import React from 'react';
import Layout from '@theme/Layout';
import ChatInterface from '../components/Chat/ChatInterface';
import clsx from 'clsx';
import styles from './chat.module.css';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function ChatPage() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`Chat - ${siteConfig.title}`}
      description="RAG Chatbot for Physical AI & Humanoid Robotics Textbook">
      <main className={styles.chatPage}>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--12">
              <div className={styles.chatWrapper}>
                <ChatInterface />
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default ChatPage;