import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'tolstoy', text: 'Здравствуйте, друг мой. О чем вы желаете побеседовать сегодня? Мои мысли и письма к вашим услугам.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', text: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMessage,
          history: messages.slice(-10) // Send last 10 messages for context
        }),
      });

      if (!response.ok) throw new Error('Ошибка связи с великим мыслителем.');

      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'tolstoy', text: data.response }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'tolstoy', text: `Простите, мои мысли запутались... (${err.message})` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="portrait-container">
        <img src="/tolstoy.png" alt="Лев Николаевич Толстой" />
      </div>
      
      <h1>Лев Николаевич Толстой</h1>
      <p style={{ fontStyle: 'italic', marginBottom: '1.5rem', opacity: 0.8 }}>
        Чат-бот на основе дневников и писем (Yandex GPT RAG)
      </p>

      <div className="chat-box">
        <div className="messages">
          {messages.map((msg, index) => {
            const hasFootnotes = msg.role === 'tolstoy' && msg.text.includes('Сноски:');
            let mainText = msg.text;
            let footnotes = '';

            if (hasFootnotes) {
              const parts = msg.text.split('Сноски:');
              mainText = parts[0];
              footnotes = parts[1];
            }

            return (
              <div key={index} className={`message ${msg.role}`}>
                <div className="main-text">{mainText}</div>
                {hasFootnotes && (
                  <>
                    <hr className="footnotes-divider" />
                    <div className="footnotes-section">
                      <span className="footnotes-title">Сноски:</span>
                      {footnotes.split('\n').map((fn, i) => (
                        fn.trim() && <div key={i}>{fn}</div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            );
          })}
          {loading && (
            <div className="message tolstoy" style={{ opacity: 0.5 }}>
              Размышляю...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Задайте вопрос графу Толстому..."
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()}>
            Спросить
          </button>
        </div>
      </div>
      
      <div style={{ marginTop: '2rem', fontSize: '0.9rem', opacity: 0.6 }}>
        &copy; 2026 Л.Н. Толстой • Цифровой аватар
      </div>
    </div>
  );
}

export default App;
