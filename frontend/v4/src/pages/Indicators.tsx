import React, { useEffect, useState } from 'react'

interface StockTechnical {
    ticker: string
    name: string
    price: number
    change_pct: number
    rsi_14: number
    macd_diff: number
    volume_ratio: number
    sma_20_bias: number
    signals: string[]
}

const Indicators: React.FC = () => {
    const [stocks, setStocks] = useState<StockTechnical[]>([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState<'ALL' | 'OVERSOLD' | 'MOMENTUM'>('ALL')

    useEffect(() => {
        // Fetch candidates and map to technical view
        // In reality, this endpoint should return raw indicators too.
        // Assuming the current candidate endpoint gives us enough to simulate or we fetch details.
        // For V4.1 Lite, we might need to adjust the backend to send these raw values in the list
        // or we simulate them for now based on the score.

        fetch('/api/v4/sniper/candidates?limit=100')
            .then(res => res.json())
            .then(data => {
                // Map API response to technical view
                const enhancedData = data.map((s: any) => ({
                    ticker: s.ticker,
                    name: s.name,
                    price: s.price,
                    change_pct: s.change_percent,
                    rsi_14: s.rsi_14 || 50,
                    macd_diff: s.macd_diff || 0,
                    volume_ratio: s.volume_ratio || 1,
                    sma_20_bias: s.sma_20_bias || 0, // Assuming this will come from API or default to 0
                    signals: s.signals || []
                }))
                setStocks(enhancedData)
                setLoading(false)
            })
            .catch(err => {
                console.error("Scanner fetch error:", err)
                setLoading(false)
            })
    }, [])

    const filteredStocks = stocks.filter(s => {
        if (filter === 'OVERSOLD') return s.rsi_14 < 45
        if (filter === 'MOMENTUM') return s.rsi_14 > 60 && s.macd_diff > 0
        return true
    })

    if (loading) return <div className="p-6 text-dark-muted">Loading Technical Scanner...</div>

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-white">Technical Scanner</h2>
                <div className="flex bg-dark-card rounded-lg p-1 border border-dark-border">
                    {(['ALL', 'OVERSOLD', 'MOMENTUM'] as const).map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${filter === f ? 'bg-sniper-green text-dark-bg' : 'text-dark-muted hover:text-white'
                                }`}
                        >
                            {f}
                        </button>
                    ))}
                </div>
            </div>

            <div className="bg-dark-card rounded-xl border border-dark-border shadow-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-dark-border/30 text-dark-muted uppercase text-xs">
                            <tr>
                                <th className="p-4">Ticker</th>
                                <th className="p-4 text-right">Price</th>
                                <th className="p-4 text-right">Change</th>
                                <th className="p-4 text-right">RSI (14)</th>
                                <th className="p-4 text-right">MACD Diff</th>
                                <th className="p-4 text-right">Vol Ratio</th>
                                <th className="p-4">Signals</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-dark-border">
                            {filteredStocks.map((stock) => (
                                <tr key={stock.ticker} className="hover:bg-dark-border/30 transition-colors group">
                                    <td className="p-4 font-medium text-white">
                                        {stock.ticker} <span className="text-dark-muted font-normal ml-1">{stock.name}</span>
                                    </td>
                                    <td className="p-4 text-right text-white">${stock.price.toFixed(2)}</td>
                                    <td className={`p-4 text-right font-bold ${stock.change_pct >= 0 ? 'text-sniper-green' : 'text-red-500'}`}>
                                        {stock.change_pct >= 0 ? '+' : ''}{stock.change_pct.toFixed(2)}%
                                    </td>
                                    <td className="p-4 text-right">
                                        <span className={`px-2 py-1 rounded ${stock.rsi_14 > 70 ? 'bg-red-500/20 text-red-500' :
                                            stock.rsi_14 < 30 ? 'bg-sniper-green/20 text-sniper-green' : 'text-dark-muted'
                                            }`}>
                                            {stock.rsi_14}
                                        </span>
                                    </td>
                                    <td className={`p-4 text-right ${stock.macd_diff > 0 ? 'text-sniper-green' : 'text-red-500'}`}>
                                        {stock.macd_diff.toFixed(2)}
                                    </td>
                                    <td className="p-4 text-right text-white">
                                        {stock.volume_ratio.toFixed(1)}x
                                    </td>
                                    <td className="p-4">
                                        <div className="flex gap-1 flex-wrap">
                                            {stock.signals.slice(0, 2).map((sig, i) => (
                                                <span key={i} className="px-2 py-0.5 bg-dark-border rounded text-[10px] text-dark-muted border border-dark-border group-hover:border-dark-muted/50">
                                                    {sig}
                                                </span>
                                            ))}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

export default Indicators
