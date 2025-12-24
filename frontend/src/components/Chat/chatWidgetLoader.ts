import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

if (ExecutionEnvironment.canUseDOM) {
  // Dynamically import and render the floating chat widget
  import('./FloatingChatWidget').then(({ default: FloatingChatWidget }) => {
    // Create a container for the widget
    const widgetContainer = document.createElement('div');
    widgetContainer.id = 'floating-chat-widget';
    document.body.appendChild(widgetContainer);

    // Render the widget using React DOM
    const React = require('react');
    const ReactDOM = require('react-dom/client');

    const root = ReactDOM.createRoot(widgetContainer);
    root.render(React.createElement(FloatingChatWidget));
  });
}