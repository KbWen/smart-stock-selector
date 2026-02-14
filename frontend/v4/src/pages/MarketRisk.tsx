import React, { useEffect, useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import { Activity, Thermometer, TrendingUp, TrendingDown } from 'lucide-react'

interface MarketStatus {
    timestamp: string
    bull_count: number
    bear_count: number
    market_temperature: number // 0-100
    status: "BULLISH" | "BEARISH" | "NEUTRAL"
    top_sector: string
}

const COLORS = ['#00ff9d', '#ef4444', '#f59e0b']

const MarketRisk: React.FC = () => {
    // Simulated data for now if API endpoint doesn't return this exact structure
    // In a real scenario, fetch from /api/market_status
    const [data, setData] = useState<MarketStatus | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/api/market_status')
            .then(res => res.json())
            .then(data => {
                // If API returns null/empty, fallbacks are handled in render or here
                setData(data)
                setLoading(false)
            })
            .catch(err => {
                console.error("Market Risk fetch error:", err)
                setLoading(false)
            })
    }, [])

    if (loading) return <div className="p-6 text-dark-muted">Scanning Market Risk...</div>
    if (!data) return null

    const pieData = [
        { name: 'Bullish', value: data.bull_count },
        { name: 'Bearish', value: data.bear_count },
    ]

    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <Activity className="text-sniper-green" />
                Market Radar
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Market Breadth */}
                <div className="bg-dark-card p-6 rounded-xl border border-dark-border shadow-lg">
                    <h3 className="text-lg font-semibold text-white mb-4">Bull/Bear Ratio</h3>
                    <div className="h-[250px] relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                            <span className="text-3xl font-bold text-white">
                                {((data.bull_count / (data.bull_count + data.bear_count)) * 100).toFixed(0)}%
                            </span>
                            <span className="text-xs text-dark-muted">BULLS</span>
                        </div>
                    </div>
                </div>

                {/* Market Temperature */}
                <div className="bg-dark-card p-6 rounded-xl border border-dark-border shadow-lg flex flex-col justify-center items-center text-center">
                    <h3 className="text-lg font-semibold text-white mb-6">Market Temperature</h3>

                    <div className="relative w-48 h-48 flex items-center justify-center mb-4">
                        {/* Simple Gauge Visualization using SVG */}
                        <svg className="w-full h-full" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#333" strokeWidth="10" />
                            <circle
                                cx="50"
                                cy="50"
                                r="45"
                                fill="none"
                                stroke={data.market_temperature > 60 ? '#ef4444' : data.market_temperature < 30 ? '#3b82f6' : '#00ff9d'}
                                strokeWidth="10"
                                strokeDasharray={`${data.market_temperature * 2.8} 280`}
                                transform="rotate(-90 50 50)"
                                strokeLinecap="round"
                            />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <Thermometer className={data.market_temperature > 60 ? 'text-red-500' : 'text-sniper-green'} size={32} />
                            <span className="text-4xl font-bold text-white mt-1">{data.market_temperature}°F</span>
                        </div>
                    </div>

                    <div className="space-y-1">
                        <div className="text-2xl font-bold text-sniper-gold tracking-widest">{data.status}</div>
                        <div className="text-sm text-dark-muted">Sector Leader: {data.top_sector}</div>
                    </div>
                </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-dark-card p-4 rounded-lg border border-dark-border text-center">
                    <div className="text-dark-muted text-xs uppercase">VIX Index</div>
                    <div className="text-xl font-bold text-white mt-1">18.45</div>
                    <div className="text-xs text-red-400 mt-1 flex justify-center items-center gap-1">
                        <TrendingUp size={12} /> +1.2%
                    </div>
                </div>
                <div className="bg-dark-card p-4 rounded-lg border border-dark-border text-center">
                    <div className="text-dark-muted text-xs uppercase">USD/TWD</div>
                    <div className="text-xl font-bold text-white mt-1">31.25</div>
                    <div className="text-xs text-sniper-green mt-1 flex justify-center items-center gap-1">
                        <TrendingDown size={12} /> -0.1%
                    </div>
                </div>
                <div className="bg-dark-card p-4 rounded-lg border border-dark-border text-center">
                    <div className="text-dark-muted text-xs uppercase">Volume (Est)</div>
                    <div className="text-xl font-bold text-white mt-1">3.2T</div>
                    <div className="text-xs text-sniper-green mt-1">Normal</div>
                </div>
                <div className="bg-dark-card p-4 rounded-lg border border-dark-border text-center">
                    <div className="text-dark-muted text-xs uppercase">Put/Call</div>
                    <div className="text-xl font-bold text-white mt-1">0.85</div>
                    <div className="text-xs text-sniper-green mt-1">Bullish</div>
                </div>
            </div>
        </div>
    )
}

export default MarketRisk
