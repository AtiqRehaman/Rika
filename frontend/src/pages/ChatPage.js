import React, { useState, useEffect, useContext, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useNavigate } from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';
import { AuthContext } from '../context/AuthContext';
// import botImage from '../airobot.png';
import { FaPaperPlane } from 'react-icons/fa';
import rikaLogo from '../rika_logo.png';

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Serif+JP:wght@600;700&display=swap');

  body {
    margin: 0;
    padding: 0;
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #e6e8ef;
    height: 100vh;
  }

  * {
    box-sizing: border-box;
  }

  .code-container {
  background: #1e1e1e;
  border-radius: 10px;
  margin-top: 8px;
  overflow: hidden;
  font-family: monospace;
}

.code-header {
  background: #2d2d2d;
  color: #aaa;
  padding: 6px 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}

.copy-btn {
  background: #444;
  color: white;
  border: none;
  padding: 4px 8px;
  border-radius: 5px;
  cursor: pointer;
}

.copy-btn:hover {
  background: #666;
}

pre {
  margin: 0;
  padding: 10px;
  overflow-x: auto;
}

`;

const Container = styled.div`
  display: flex;
  width: 100%;
  max-width: 1800px;
  height: 90vh;
  margin: auto;
  background: #161a23;
  border-radius: 18px;
  border: 1px solid #232734;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  overflow: hidden;
`;

const DesignSection = styled.div`
  flex: 1;
  background: #12141c;
  border-right: 1px solid #232734;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
`;

const BotImage = styled.img`
  width: 180px;
  height: auto;
  margin-bottom: 1.5rem;
`;

const BotName = styled.h1`
  font-family: 'Noto Serif JP', serif;
  font-size: 2.4rem;
  letter-spacing: 3px;
  margin: 0;
  color: #ffffff;
`;

// const SubTitle = styled.p`
//   margin-top: 8px;
//   color: #8a90a6;
//   font-size: 0.9rem;
// `;

const ChatSection = styled.div`
  flex: 2;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1.5rem;
`;

const ChatArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding-right: 8px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const ChatBubble = styled.div`
  background: ${(props) =>
    props.isUser ? '#4c51bf' : 'transparent'};

  color: #ffffff;
  padding: 12px 16px;
  border-radius: 16px;
  max-width: 55%;
  width: fit-content;
  word-break: break-word;
  line-height: 1.5;
  font-size: 0.95rem;
  border: 1px solid ${(props) =>
    props.isUser ? '#5a67d8' : 'transparent'};
  white-space: pre-wrap;
`;


const MessageRow = styled.div`
  display: flex;
  width: 100%;
  justify-content: ${(props) =>
    props.isUser ? 'flex-end' : 'flex-start'};
  margin-bottom: 14px;
`;


const BotAvatar = styled.img`
  width: 34px;
  height: 34px;
  border-radius: 50%;
  object-fit: cover;
  margin-top: 4px;
`;


const TimeStamp = styled.span`
  display: block;
  font-size: 0.7rem;
  margin-top: 6px;
  color: #8a90a6;
`;

const FormContainer = styled.form`
  display: flex;
  gap: 10px;
  align-items: center;
`;

const InputField = styled.textarea`
  flex: 1;
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid #2c3242;
  background: #1a1f2b;
  font-size: 0.95rem;
  color: #e6e8ef;
  outline: none;
  resize: none;
  min-height: 42px;
  max-height: 120px;
  line-height: 1.5;
  overflow-y: auto;

  &:focus {
    border-color: #5a67d8;
  }
`;


const SubmitButton = styled.button`
  background: #4c51bf;
  color: white;
  padding: 10px 14px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: 0.2s ease;

  &:hover {
    background: #5a67d8;
  }
`;

const ChatPage = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const { user, isAuth } = useContext(AuthContext);
  const navigate = useNavigate();
  const chatAreaRef = useRef(null);
  const email = user ? user.email : null;

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await axios.get(
          `http://localhost:5000/api/chats?email=${email}`,
          {
            headers: {
              'x-api-key': process.env.REACT_APP_API_KEY,
            },
            withCredentials: true,
          }
        );

        if (response.data && response.data.messages) {
          const chatHistory = response.data.messages.flatMap((msg) => [
            {
              content: msg.content,
              isUser: true,
              timestamp: new Date(msg.timestamp),
            },
            {
              content: msg.response,
              isUser: false,
              timestamp: new Date(msg.timestamp),
            },
          ]);

          setMessages(
            chatHistory.sort((a, b) => a.timestamp - b.timestamp)
          );
        }
      } catch (err) {
        console.error('Error fetching chat history:', err);
      }
    };

    if (email) fetchMessages();
  }, [email]);

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop =
        chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

const renderMessage = (content) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");

          if (!inline) {
            return (
              <div className="code-container">
                <div className="code-header">
                  <span>{match ? match[1] : "code"}</span>
                  <button
                    onClick={() =>
                      navigator.clipboard.writeText(String(children))
                    }
                    className="copy-btn"
                  >
                    Copy
                  </button>
                </div>
                <pre>
                  <code {...props}>{children}</code>
                </pre>
              </div>
            );
          }

          return <code className="inline-code">{children}</code>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};


  const handleSubmit = async (e) => {
  e.preventDefault();
  if (!input.trim()) return;

  const userMessage = {
    content: input,
    isUser: true,
    timestamp: new Date(),
  };

  setMessages((prev) => [...prev, userMessage]);

  const currentInput = input;
  setInput(""); // clear input immediately

  try {
    const response = await fetch(
      "https://compression-rats-meditation-effective.trycloudflare.com/chat",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: "atiq_session",
          message: currentInput,
        }),
      }
    );

    if (!response.body) throw new Error("No response body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let botMessage = {
      content: "",
      isUser: false,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, botMessage]);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      botMessage.content += chunk;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { ...botMessage };
        return updated;
      });
    }
  } catch (err) {
    setMessages((prev) => [
      ...prev,
      {
        content: "An error occurred. Please try again later.",
        isUser: false,
        timestamp: new Date(),
      },
    ]);
  }
};


  return (
    <>
      <GlobalStyle />
      <Container>
        {/* <DesignSection>
          <BotImage src={botImage} alt="Rika AI" />
          <BotName>Rika</BotName>
          <SubTitle>里香 • AI Coding Assistant</SubTitle>
        </DesignSection> */}

        <ChatSection>
          <ChatArea ref={chatAreaRef}>
            {messages.map((message, index) => (
  <MessageRow key={index} isUser={message.isUser}>

    {!message.isUser && (
      <BotAvatar src={rikaLogo} alt="R" />
    )}

    <ChatBubble isUser={message.isUser}>
      {renderMessage(message.content)}
      <TimeStamp>
        {new Date(message.timestamp).toLocaleTimeString()}
      </TimeStamp>
    </ChatBubble>

  </MessageRow>
))}

          </ChatArea>

          <FormContainer onSubmit={handleSubmit}>
            <InputField
              placeholder="Ask Rika something..."
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = e.target.scrollHeight + 'px';
              }}

              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />

            <SubmitButton type="submit">
              <FaPaperPlane />
            </SubmitButton>
          </FormContainer>
        </ChatSection>
      </Container>
    </>
  );
};

export default ChatPage;
