import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import DataExample from './components/DataExample'
import AIMonitoring from './components/AIMonitoring'
import './components/DataExample.css'
import './components/AIMonitoring.css'

function App() {
  const [showDataExample, setShowDataExample] = useState(false)
  const [showAIMonitoring, setShowAIMonitoring] = useState(false)

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Sailo - AI ETL Pipeline</h1>
      <div className="card">
        <button onClick={() => setShowDataExample(!showDataExample)}>
          {showDataExample ? 'Hide' : 'Show'} Database Example
        </button>
        <button onClick={() => setShowAIMonitoring(!showAIMonitoring)}>
          {showAIMonitoring ? 'Hide' : 'Show'} AI Monitoring
        </button>
        <p>
          Configure your Supabase credentials in <code>.env.local</code> to get started
        </p>
      </div>
      
      {showDataExample && <DataExample />}
      {showAIMonitoring && <AIMonitoring />}
      
      <p className="read-the-docs">
        Test your PostgreSQL data retrieval and AI-powered monitoring pipeline
      </p>
    </>
  )
}

export default App
