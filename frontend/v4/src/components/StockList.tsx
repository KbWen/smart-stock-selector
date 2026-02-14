import React, { useEffect, useState } from 'react'

interface StockCandidate {
    ticker: string
    name: string
    price: number
    rise_score: number
    ai_prob: number
}

interface StockListProps {
    onSelect: (ticker: string) => void
}

const StockList: React.FC<StockListProps> = ({ onSelect }) => {
    const [stocks, setStocks] = useState<StockCandidate[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/api/v4/sniper/candidates?limit=50')
            .then(res => res.json())
            .then(data => {
                setStocks(data)
                setLoading(false)
            })
            .catch(err => {
                console.error("Failed to fetch candidates:", err)
                setLoading(false)
            })
    }, [])

    if (loading) {
        return <div className="p-6 text-center text-dark-muted">Loading Sniper Candidates...</div>
    }

    if (stocks.length === 0) {
        return <div className="p-6 text-center text-dark-muted">No candidates found. Try running sync.</div>
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
                <thead>
                    <tr className="text-dark-muted border-b border-dark-border">
                        <th className="p-3">Ticker</th>
                        <th className="p-3">Price</th>
                        <th className="p-3">Rise Score</th>
                        <th className="p-3">AI Prob</th>
                        <th className="p-3">Action</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                    {stocks.map((stock) => (
                        <tr key={stock.ticker} className="hover:bg-dark-border/30 transition-colors">
                            <td className="p-3">
                                <span className="font-medium text-white block">{stock.ticker}</span>
                                <span className="text-xs text-dark-muted">{stock.name}</span>
                            </td>
                            <td className="p-3 text-dark-text">${stock.price.toFixed(2)}</td>
                            <td className="p-3">
                                <span className={`font-bold ${stock.rise_score >= 80 ? 'text-sniper-green' : 'text-white'}`}>
                                    {stock.rise_score}
                                </span>
                            </td>
                            <td className="p-3">
                                <span className={`font-bold ${stock.ai_prob >= 70 ? 'text-sniper-gold' : 'text-dark-muted'}`}>
                                    {stock.ai_prob}%
                                </span>
                            </td>
                            <td className="p-3">
                                <button
                                    onClick={() => onSelect(stock.ticker)}
                                    className="text-xs bg-dark-border hover:bg-white/10 px-3 py-1.5 rounded text-white transition-colors border border-transparent hover:border-dark-muted"
                                >
                                    Analyze
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

export default StockList
