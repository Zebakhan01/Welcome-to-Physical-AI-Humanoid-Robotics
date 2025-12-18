import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <h1 className={styles.welcomeTitle}>Welcome to Physical AI & Humanoid Robotics</h1>
        <p className={styles.welcomeSubtitle}>
          An AI-native textbook for Physical AI, Robotics, ROS2, Simulation, and Vision-Language-Action systems.
          Explore cutting-edge concepts in humanoid robotics and intelligent physical systems.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro/intro">
            Start the Textbook
          </Link>
          <Link
            className="button button--primary button--lg"
            to="/chat"
            style={{ marginLeft: '1rem' }}>
            Ask AI Assistant
          </Link>
        </div>
      </div>
    </header>
  );
}

function Card({ title, description, to, bgClass }) {
  return (
    <Link to={to} className={`${styles.card} ${bgClass}`}>
      <div className={styles.cardHoverIndicator}>→</div>
      <div className={styles.cardContent}>
        <h3 className={styles.cardTitle}>{title}</h3>
        <p className={styles.cardDescription}>{description}</p>
      </div>
    </Link>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics: An AI-Native Textbook for Advanced Robotics Education">
      <HomepageHeader />
      <main>
        <section className={styles.cardGrid}>
          <Card
            title="ROS 2"
            description="Robot Operating System for building robotic applications with advanced communication and control frameworks."
            to="/docs/modules/ros-2"
            bgClass="bg-ros"
          />
          <Card
            title="Gazebo"
            description="3D simulation environment for robotics research and development with realistic physics."
            to="/docs/modules/gazebo"
            bgClass="bg-gazebo"
          />
          <Card
            title="Unity"
            description="Real-time 3D development platform for creating immersive robotics simulations and interfaces."
            to="/docs/modules/unity"
            bgClass="bg-unity"
          />
          <Card
            title="NVIDIA Isaac"
            description="GPU-accelerated robotics platform for AI-powered perception and navigation systems."
            to="/docs/modules/nvidia-isaac"
            bgClass="bg-isaac"
          />
          <Card
            title="Vision–Language–Action (VLA)"
            description="Multimodal AI systems that integrate visual perception, language understanding, and physical action."
            to="/docs/modules/vla"
            bgClass="bg-vla"
          />
        </section>
      </main>
    </Layout>
  );
}