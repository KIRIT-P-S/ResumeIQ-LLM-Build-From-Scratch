import React, { useState, useRef, useEffect } from 'react';
import {
  Search, Send, FileText, Cpu, ChevronDown, ChevronUp,
  Sparkles, BookOpen, AlertCircle, CheckCircle, Clock,
  Zap, Users, User, ToggleLeft, ToggleRight
} from 'lucide-react';
import './App.css';

const API = 'http://192.168.4.99:8002';

const JD_SUGGESTIONS = [
  'We need a Python developer with ML experience and React skills',
  'Looking for a Java Spring Boot backend developer with SQL knowledge',
  'Full stack developer with Node.js, React and MongoDB experience',
  'Data analyst with Python, SQL and Power BI skills',
  'Android developer with Kotlin and Firebase experience',
];

const SINGLE_SUGGESTIONS = [
  'What programming languages does Abhinav know?',
  'What projects has the candidate built?',
  'What is the candidate\'s CGPA?',
  'What certifications does the candidate have?',
  'What is the candidate\'s educational background?',
];
const MULTI_SUGGESTIONS = [
  'Find candidates with Python skills',
  'Who knows React and Node.js?',
  'Find candidates with machine learning experience',
  'Who has worked with Spring Boot?',
  'Find candidates with SQL knowledge',
];

function TypingDots() {
  return <div className="typing-dots"><span /><span /><span /></div>;
}

function SourceBadge({ name }) {
  const short = name.replace(/synthetic_resume_0*/, 'Resume #').replace('.txt','').replace('.pdf','');
  return <span className="source-badge"><FileText size={11}/>{short}</span>;
}

function CandidateCard({ c, index }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(c.score * 100);
  const name = c.filename.replace(/\.pdf$/i,'').replace(/\.txt$/i,'').replace(/_/g,' ');
  if (c.status === 'NOT_FOUND') return null;
  return (
    <div className="candidate-card">
      <div className="candidate-header">
        <div className="candidate-rank">#{index + 1}</div>
        <div className="candidate-info">
          <div className="candidate-name">{name}</div>
          <div className="candidate-folder">{c.student_folder}</div>
        </div>
        <div className="candidate-score">
          <div className="score-bar-wrap">
            <div className="score-bar" style={{width: `${Math.min(pct * 2, 100)}%`}} />
          </div>
          <span>{pct}% match</span>
        </div>
      </div>
      <div className="candidate-answer">
        <span className="found-badge">FOUND</span>{c.answer}
      </div>
      <button className="ctx-toggle" onClick={() => setOpen(v => !v)}>
        <BookOpen size={12}/>
        {open ? 'Hide context' : 'Show context'}
        {open ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
      </button>
      {open && (
        <div className="context-box">
          <div className="context-label">Retrieved context</div>
          <pre>{c.context}</pre>
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  const [showCtx, setShowCtx] = useState(false);
  const isUser = msg.role === 'user';
  if (isUser) return (
    <div className="message message-user">
      <div className="message-body">
        <div className="bubble bubble-user">{msg.content}</div>
        <div className="message-time">{msg.time}</div>
      </div>
      <div className="user-avatar">You</div>
    </div>
  );
  return (
    <div className="message message-ai">
      <div className="ai-avatar"><Cpu size={16}/></div>
      <div className="message-body">
        {msg.loading ? (
          <div className="bubble bubble-ai"><TypingDots/></div>
        ) : (
          <>
            {msg.candidates?.length > 0 ? (
              <div className="multi-results">
                <div className="multi-header">
                  <Users size={15}/> {msg.content}
                </div>
                {msg.candidates.map((c, i) => <CandidateCard key={i} c={c} index={i}/>)}
              </div>
            ) : (
              <div className={`bubble bubble-ai ${msg.status==='NOT_FOUND'?'bubble-not-found':''}`}>
                {msg.status === 'NOT_FOUND' && <span className="not-found-badge">NOT FOUND</span>}
                {msg.status === 'FOUND' && <span className="found-badge">FOUND</span>}
                {msg.content}
              </div>
            )}
            {msg.sources?.length > 0 && msg.candidates?.length === 0 && (
              <div className="message-meta">
                <div className="sources-row">
                  {msg.sources.map((s,i) => <SourceBadge key={i} name={s}/>)}
                </div>
                {msg.context && (
                  <button className="ctx-toggle" onClick={() => setShowCtx(v => !v)}>
                    <BookOpen size={12}/>
                    {showCtx ? 'Hide context' : 'Show context'}
                    {showCtx ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
                  </button>
                )}
              </div>
            )}
            {showCtx && (
              <div className="context-box">
                <div className="context-label">Retrieved context</div>
                <pre>{msg.context}</pre>
              </div>
            )}
          </>
        )}
        <div className="message-time">{msg.time}</div>
      </div>
    </div>
  );
}

function StatusBar({ status, count }) {
  const icon = status === 'ok'
    ? <CheckCircle size={13} className="status-green"/>
    : status === 'error'
    ? <AlertCircle size={13} className="status-red"/>
    : <Clock size={13} className="status-yellow"/>;
  const label = status === 'ok' ? `API connected · ${count} resumes` : status === 'error' ? 'API offline' : 'Connecting…';
  return (
    <div className={`status-bar status-${status}`}>
      {icon}<span>{label}</span>
      <span className="status-model"><Zap size={11}/>Scratch SFT v3</span>
    </div>
  );
}

export default function App() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState('');
  const [loading, setLoading]     = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [resumeCount, setResumeCount] = useState(0);
  const [topK, setTopK]           = useState(5);
  const [temp, setTemp]           = useState(0.3);
  const [mode, setMode]           = useState('single'); // 'single' | 'multi' | 'jd'
  const [showSettings, setShowSettings] = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => { setApiStatus('ok'); setResumeCount(d.resumes_indexed || 0); })
      .catch(() => setApiStatus('error'));
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Auto-detect if question is a multi-candidate search
  const sendJD = async (jd) => {
    const q = (jd || input).trim();
    if (!q || loading) return;
    setInput('');
    setMessages(prev => [...prev,
      { role: 'user', content: q, time: now(), isJD: true },
      { role: 'ai', loading: true, time: now() }
    ]);
    setLoading(true);
    try {
      const res = await fetch(`${API}/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd: q, top_k: topK, explain: false }),
      });
      const data = await res.json();
      setMessages(prev => [...prev.slice(0, -1), {
        role: 'ai', content: `Found ${data.total_candidates} matching candidates.`,
        status: 'FOUND', sources: data.candidates.map(c => c.filename),
        context: '', candidates: data.candidates.map(c => ({
          filename: c.filename, student_folder: c.student_folder,
          status: 'FOUND', answer: c.explanation,
          context: c.context,
          score: c.score,
        })), time: now()
      }]);
    } catch {
      setMessages(prev => [...prev.slice(0, -1), {
        role: 'ai', content: '⚠️ Could not reach the API.',
        sources: [], candidates: [], time: now()
      }]);
    }
    setLoading(false);
    inputRef.current?.focus();
  };

  const isMultiQuery = (q) => {
    const lower = q.toLowerCase();
    return /^(find|list|show|give me|who|which candidates?|search)/.test(lower) ||
           lower.includes('candidates with') || lower.includes('resumes with') ||
           lower.includes('students with') || lower.includes('who knows') ||
           lower.includes('who has') || lower.includes('who have');
  };

  const send = async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;
    const effectiveMode = isMultiQuery(q) ? 'multi' : mode;
    setInput('');
    setMessages(prev => [...prev,
      { role: 'user', content: q, time: now() },
      { role: 'ai', loading: true, time: now() }
    ]);
    setLoading(true);
    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: topK, mode: effectiveMode }),
      });
      const data = await res.json();
      setMessages(prev => [...prev.slice(0, -1), {
        role: 'ai', content: data.answer, status: data.status,
        sources: data.sources, context: data.context_used,
        candidates: data.candidates || [], time: now()
      }]);
    } catch {
      setMessages(prev => [...prev.slice(0, -1), {
        role: 'ai', content: '⚠️ Could not reach the API.',
        sources: [], candidates: [], time: now()
      }]);
    }
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleKey = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };
  const suggestions = mode === 'jd' ? JD_SUGGESTIONS : mode === 'multi' ? MULTI_SUGGESTIONS : SINGLE_SUGGESTIONS;
  const handleSend = mode === 'jd' ? sendJD : send;
  const isEmpty = messages.length === 0;

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon"><Sparkles size={20}/></div>
          <div>
            <div className="logo-title">ResumeAI</div>
            <div className="logo-sub">Scratch LLM · RAG</div>
          </div>
        </div>

        <StatusBar status={apiStatus} count={resumeCount}/>

        {/* Mode toggle */}
        <div className="mode-toggle-wrap">
          <button className={`mode-btn ${mode==='single'?'active':''}`} onClick={() => setMode('single')}>
            <User size={14}/> Single
          </button>
          <button className={`mode-btn ${mode==='multi'?'active':''}`} onClick={() => setMode('multi')}>
            <Users size={14}/> Multi-candidate
          </button>
          <button className={`mode-btn ${mode==='jd'?'active':''}`} onClick={() => setMode('jd')} style={{marginTop:'6px',flex:'unset',width:'100%'}}>
            <Sparkles size={14}/> JD Match
          </button>
        </div>
        <div className="mode-hint">
          {mode === 'jd'
            ? 'Paste a Job Description — encoder + decoder ranks all 440 resumes.'
            : mode === 'multi'
            ? 'Search across all resumes and rank candidates by relevance.'
            : 'Ask a specific question about the best matching resume.'}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Suggested questions</div>
          {suggestions.map((s, i) => (
            <button key={i} className="suggestion-btn" onClick={() => handleSend(s)}>
              <Search size={13}/>{s}
            </button>
          ))}
        </div>

        <div className="sidebar-section">
          <button className="settings-toggle" onClick={() => setShowSettings(v => !v)}>
            <span>Settings</span>
            {showSettings ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
          </button>
          {showSettings && (
            <div className="settings-panel">
              <label>
                <span>{mode === 'multi' ? 'Top candidates' : 'Top-K chunks'}: <b>{topK}</b></span>
                <input type="range" min={1} max={10} value={topK} onChange={e => setTopK(+e.target.value)}/>
              </label>
              <label>
                <span>Temperature: <b>{temp}</b></span>
                <input type="range" min={0} max={1} step={0.05} value={temp} onChange={e => setTemp(+e.target.value)}/>
              </label>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          201M-param transformer trained from scratch on 1B tokens. SFT v3 on 180K resume QA examples.
        </div>
      </aside>

      {/* Chat */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="chat-header-title">
            <Cpu size={18} className="header-icon"/>
            Resume Question Answering
          </div>
          <div className="chat-header-sub">
            {resumeCount > 0 ? `${resumeCount} resumes indexed` : 'Ask anything about the indexed resumes'}
          </div>
        </header>

        <div className="messages-container">
          {isEmpty ? (
            <div className="empty-state">
              <div className="empty-icon"><Sparkles size={40}/></div>
              <h2>Ask about any resume</h2>
              <p>Use <b>Single</b> mode to ask about a specific candidate, <b>Multi-candidate</b> to search all {resumeCount} resumes, or <b>JD Match</b> to paste a job description and rank candidates.</p>
              <div className="empty-chips">
                {suggestions.slice(0, 3).map((s, i) => (
                  <button key={i} className="empty-chip" onClick={() => handleSend(s)}>{s}</button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => <Message key={i} msg={msg}/>)
          )}
          <div ref={bottomRef}/>
        </div>

        <div className="input-area">
          <div className="input-box">
            <div className={`mode-pill mode-pill-${mode}`}>
              {mode === 'multi' ? <Users size={12}/> : <User size={12}/>}
              {mode === 'multi' ? 'Multi' : 'Single'}
            </div>
            <textarea
              ref={inputRef} rows={1}
              placeholder={mode === 'jd' ? 'Paste job description here…' : mode === 'multi' ? 'Find candidates with Python skills…' : 'Ask about a specific resume…'}
              value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey} disabled={loading}
            />
            <button className={`send-btn ${loading?'send-loading':''}`}
              onClick={() => handleSend()} disabled={loading || !input.trim()}>
              <Send size={18}/>
            </button>
          </div>
          <div className="input-hint">Enter to send · Shift+Enter for new line</div>
        </div>
      </main>
    </div>
  );
}
