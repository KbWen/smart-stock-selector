import React, { useEffect, useState } from 'react'

interface SniperCardProps {
    ticker: string | null
}

interface StockDetail {
    ticker: string
    name: string
    price: number
    rise_score_breakdown: {
        total: number
        trend: number
        momentum: number
        volatility: number
    }
    ai_probability: number
    analyst_summary: string
    signals: {
        squeeze: boolean
        golden_cross: boolean
        volume_spike: boolean
    }
}

const SniperCard: React.FC<SniperCardProps> = ({ ticker }) => {
    const [data, setData] = useState<StockDetail | null>(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        if (!ticker) return

        setLoading(true)
        fetch(`/api/v4/stock/${ticker}`)
            .then(res => res.json())
            .then(data => {
                setData(data)
                setLoading(false)
            })
            .catch(err => {
                console.error("Failed to fetch stock detail:", err)
                setLoading(false)
            })
    }, [ticker])

    if (!ticker) {
        return (
            <div className="bg-dark-card border border-dark-border rounded-xl p-8 shadow-xl flex items-center justify-center text-dark-muted h-full min-h-[300px]">
                <p>Select a stock to view detailed analysis</p>
            </div>
        )
    }

    if (loading) {
        return (
            <div className="bg-dark-card border border-dark-border rounded-xl p-8 shadow-xl flex items-center justify-center text-dark-muted h-full min-h-[300px]">
                <p className="animate-pulse">Running AI Analysis for {ticker}...</p>
            </div>
        )
    }

    if (!data) return null

    return (
        <div className="bg-dark-card border border-dark-border rounded-xl p-6 shadow-xl sticky top-24">
            <div className="flex justify-between items-start mb-4">
                <div>
                    <h2 className="text-3xl font-bold text-white tracking-tight">{data.ticker}</h2>
                    <p className="text-dark-muted text-sm">{data.name}</p>
                </div>
                <div className="flex flex-col items-end">
                    <div className="text-2xl font-bold text-white">${data.price.toFixed(2)}</div>
                    <div className="bg-sniper-green/10 text-sniper-green px-3 py-1 rounded-full text-xs font-bold border border-sniper-green/20 mt-1">
                        STRONG BUY
                    </div>
                </div>
            </div>

            <div className="space-y-6">
                {/* Score Breakdown */}
                <div className="grid grid-cols-2 gap-4 bg-dark-bg/50 p-4 rounded-lg border border-dark-border">
                    <div>
                        <div className="text-xs text-dark-muted uppercase tracking-wider">Total Rise Score</div>
                        <div className="text-2xl font-bold text-sniper-green">{data.rise_score_breakdown.total}</div>
                    </div>
                    <div>
                        <div className="text-xs text-dark-muted uppercase tracking-wider">AI Probability</div>
                        <div className="text-2xl font-bold text-sniper-gold">{data.ai_probability}%</div>
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="bg-dark-border/20 p-2 rounded">
                        <span className="block text-dark-muted">Trend</span>
                        <span className="font-bold text-white">{data.rise_score_breakdown.trend}</span>
                    </div>
                    <div className="bg-dark-border/20 p-2 rounded">
                        <span className="block text-dark-muted">Momentum</span>
                        <span className="font-bold text-white">{data.rise_score_breakdown.momentum}</span>
                    </div>
                    <div className="bg-dark-border/20 p-2 rounded">
                        <span className="block text-dark-muted">Volatility</span>
                        <span className="font-bold text-white">{data.rise_score_breakdown.volatility}</span>
                    </div>
                </div>

                <div className="border-t border-dark-border pt-4">
                    <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                        <span>🤖</span> AI Analyst Insight
                    </h3>
                    <div className="text-sm text-dark-muted leading-relaxed whitespace-pre-line">
                        {data.analyst_summary}
                    </div>
                </div>

                {/* Signals Badges */}
                <div className="flex flex-wrap gap-2 pt-2">
                    {data.signals.squeeze && (
                        <span className="px-2 py-1 bg-yellow-500/10 text-yellow-500 text-xs rounded border border-yellow-500/20">🔥 Squeeze</span>
                    )}
                    {data.signals.golden_cross && (
                        <span className="px-2 py-1 bg-blue-500/10 text-blue-500 text-xs rounded border border-blue-500/20">✨ Golden Cross</span>
                    )}
                    {data.signals.volume_spike && (
                        <span className="px-2 py-1 bg-purple-500/10 text-purple-500 text-xs rounded border border-purple-500/20">📢 Volume Spike</span>
                    )}
                </div>
            </div>
        </div>
    )
}

export default SniperCard
