import React, { useState } from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './signin.module.css';

export default function Signin() {
  const { siteConfig } = useDocusaurusContext();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission
    console.log('Form submitted:', { ...formData, isLogin });
  };

  return (
    <Layout
      title={`${isLogin ? 'Log In' : 'Sign Up'} | ${siteConfig.title}`}
      description="Sign in or create an account for the Physical AI & Humanoid Robotics textbook">
      <main className={clsx('container', styles.signinPage)}>
        <div className={styles.signinContainer}>
          <div className={styles.logoSection}>
            <img
              src="/img/logo.jpg"
              alt="Physical AI & Humanoid Robotics Logo"
              className={styles.logo}
            />
            <h1 className={styles.title}>{siteConfig.title}</h1>
          </div>

          <div className={styles.formContainer}>
            <div className={styles.formToggle}>
              <button
                className={clsx(styles.toggleButton, {
                  [styles.active]: isLogin
                })}
                onClick={() => setIsLogin(true)}
              >
                Log In
              </button>
              <button
                className={clsx(styles.toggleButton, {
                  [styles.active]: !isLogin
                })}
                onClick={() => setIsLogin(false)}
              >
                Sign Up
              </button>
            </div>

            <form onSubmit={handleSubmit} className={styles.form}>
              {!isLogin && (
                <div className={styles.inputGroup}>
                  <label htmlFor="name" className={styles.label}>
                    Full Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={styles.input}
                    required={!isLogin}
                  />
                </div>
              )}

              <div className={styles.inputGroup}>
                <label htmlFor="email" className={styles.label}>
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className={styles.input}
                  required
                />
              </div>

              <div className={styles.inputGroup}>
                <label htmlFor="password" className={styles.label}>
                  Password
                </label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className={styles.input}
                  required
                />
              </div>

              {!isLogin && (
                <div className={styles.inputGroup}>
                  <label htmlFor="confirmPassword" className={styles.label}>
                    Confirm Password
                  </label>
                  <input
                    type="password"
                    id="confirmPassword"
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={styles.input}
                    required={!isLogin}
                  />
                </div>
              )}

              {isLogin && (
                <div className={styles.checkboxGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      className={styles.checkbox}
                    />
                    <span className={styles.checkboxText}>Remember me</span>
                  </label>
                </div>
              )}

              <button type="submit" className={styles.submitButton}>
                {isLogin ? 'Log In' : 'Sign Up'}
              </button>

              {isLogin && (
                <div className={styles.forgotPassword}>
                  <a href="#" className={styles.link}>Forgot password?</a>
                </div>
              )}
            </form>

            <div className={styles.divider}>
              <span className={styles.dividerText}>OR</span>
            </div>

            <div className={styles.socialLogin}>
              <button className={clsx(styles.socialButton, styles.google)}>
                Continue with Google
              </button>
              <button className={clsx(styles.socialButton, styles.github)}>
                Continue with GitHub
              </button>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}