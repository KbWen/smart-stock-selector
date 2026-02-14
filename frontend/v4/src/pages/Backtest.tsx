import React, { useEffect, useState } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    Legend
} from 'recharts'

interface BacktestResult {
    date: string
    equity: number
    win_rate: number
    trades: number
}

const Backtest: React.FC = () => {
    const [data, setData] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [versions, setVersions] = useState<any[]>([])
    const [selectedVersion, setSelectedVersion] = useState<string>("latest")
    const [days, setDays] = useState<number>(30)

    useEffect(() => {
        // Fetch available models
        fetch('/api/models')
            .then(res => res.json())
            .then(data => setVersions(data))
            .catch(err => console.error("Models fetch error:", err))
    }, [])

    const runBacktest = () => {
        setLoading(true)
        fetch(`/api/backtest?days=${days}&version=${selectedVersion}`)
            .then(res => res.json())
            .then(data => {
                setData(data)
                setLoading(false)
            })
            .catch(err => {
                console.error("Backtest fetch error:", err)
                setLoading(false)
            })
    }

    useEffect(() => {
        runBacktest()
    }, [selectedVersion, days])

    if (loading && !data) return <div className="p-6 text-dark-muted">Loading Backtest Engine...</div>

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <h2 className="text-2xl font-bold text-white">AI Backtest Performance</h2>

                <div className="flex items-center gap-3">
                    <select
                        value={selectedVersion}
                        onChange={(e) => setSelectedVersion(e.target.value)}
                        className="bg-dark-card border border-dark-border text-white text-sm rounded-lg p-2 focus:ring-blue-500"
                    >
                        <option value="latest">Latest Model</option>
                        {versions.map(v => (
                            <option key={v.version} value={v.version}>{v.version} ({v.timestamp})</option>
                        ))}
                    </select>

                    <select
                        value={days}
                        onChange={(e) => setDays(Number(e.target.value))}
                        className="bg-dark-card border border-dark-border text-white text-sm rounded-lg p-2 focus:ring-blue-500"
                    >
                        <option value={10}>10 Days</option>
                        <option value={30}>30 Days</option>
                        <option value={60}>60 Days</option>
                    </select>

                    <button
                        onClick={runBacktest}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                    >
                        Run Backtest
                    </button>
                </div>
            </div>

            {data?.error ? (
                <div className="bg-red-900/20 border border-red-500/50 p-4 rounded-lg text-red-200">
                    {data.error}
                </div>
            ) : (
                <>
                    {/* Summary Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
                            <p className="text-dark-muted text-xs mb-1 uppercase tracking-wider">Avg Return</p>
                            <p className={`text-2xl font-bold ${data?.summary?.avg_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {(data?.summary?.avg_return * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
                            <p className="text-dark-muted text-xs mb-1 uppercase tracking-wider">Win Rate</p>
                            <p className="text-2xl font-bold text-blue-400">
                                {(data?.summary?.win_rate * 100).toFixed(1)}%
                            </p>
                        </div>
                        <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
                            <p className="text-dark-muted text-xs mb-1 uppercase tracking-wider">Signals Found</p>
                            <p className="text-2xl font-bold text-white">{data?.candidate_pool_size}</p>
                        </div>
                        <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
                            <p className="text-dark-muted text-xs mb-1 uppercase tracking-wider">Best Capture</p>
                            <p className="text-sm font-semibold text-green-400 truncate mt-1">
                                {data?.summary?.best_stock} (+{(data?.summary?.best_return * 100).toFixed(1)}%)
                            </p>
                        </div>
                    </div>

                    {/* Top Picks Table */}
                    <div className="bg-dark-card rounded-xl border border-dark-border overflow-hidden shadow-lg">
                        <div className="px-6 py-4 border-b border-dark-border flex justify-between items-center">
                            <h3 className="text-lg font-semibold text-white">Top 10 High Confidence Picks (Simulated Entry)</h3>
                            <span className="text-xs text-dark-muted italic">Simulated Date: {data?.simulated_date}</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead className="bg-dark-bg text-dark-muted text-xs uppercase">
                                    <tr>
                                        <th className="px-6 py-3">Ticker</th>
                                        <th className="px-6 py-3">Entry Price</th>
                                        <th className="px-6 py-3">Current</th>
                                        <th className="px-6 py-3">Return</th>
                                        <th className="px-6 py-3">AI Prob</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-dark-border text-sm">
                                    {data?.top_picks?.map((p: any) => (
                                        <tr key={p.ticker} className="hover:bg-white/5 transition-colors">
                                            <td className="px-6 py-4">
                                                <div className="font-bold text-white">{p.ticker}</div>
                                                <div className="text-xs text-dark-muted">{p.name}</div>
                                            </td>
                                            <td className="px-6 py-4 text-white">${p.entry_price.toFixed(2)}</td>
                                            <td className="px-6 py-4 text-white">${p.current_price.toFixed(2)}</td>
                                            <td className={`px-6 py-4 font-bold ${p.actual_return >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                                {(p.actual_return * 100).toFixed(2)}%
                                            </td>
                                            <td className="px-6 py-4 text-blue-400">{(p.ai_prob_at_entry * 100).toFixed(1)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    )
}

export default Backtest
