import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import GlobalLayout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Backtest from './pages/Backtest'
import MarketRisk from './pages/MarketRisk'
import Indicators from './pages/Indicators'

function App() {
  return (
    <BrowserRouter>
      <GlobalLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/risk" element={<MarketRisk />} />
          <Route path="/indicators" element={<Indicators />} />
        </Routes>
      </GlobalLayout>
    </BrowserRouter>
  )
}

export default App
