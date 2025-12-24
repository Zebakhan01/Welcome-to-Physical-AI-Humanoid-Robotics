import React from 'react';
import Layout from '@theme-original/Layout';
import FloatingChatWidget from '../components/Chat/FloatingChatWidget';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <FloatingChatWidget />
    </>
  );
}