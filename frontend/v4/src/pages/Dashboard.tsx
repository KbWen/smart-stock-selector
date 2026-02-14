import React from 'react'
import StockList from '../components/StockList'
import SniperCard from '../components/SniperCard'

const Dashboard: React.FC = () => {
    const [selectedTicker, setSelectedTicker] = React.useState<string | null>(null)

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Market Status Card Placeholder */}
                <div className="bg-dark-card p-6 rounded-xl border border-dark-border shadow-lg">
                    <h2 className="text-lg font-semibold text-dark-muted mb-2">Market Status</h2>
                    <div className="text-3xl font-bold text-sniper-green">BULLISH</div>
                    <div className="text-sm text-dark-muted mt-1">Temperature: 75°F</div>
                </div>

                {/* AI Confidence Card Placeholder */}
                <div className="bg-dark-card p-6 rounded-xl border border-dark-border shadow-lg">
                    <h2 className="text-lg font-semibold text-dark-muted mb-2">AI Confidence</h2>
                    <div className="text-3xl font-bold text-sniper-gold">71.4%</div>
                    <div className="text-sm text-dark-muted mt-1">Model v4.1 (Lite)</div>
                </div>

                {/* Action Card Placeholder */}
                <div className="bg-dark-card p-6 rounded-xl border border-dark-border shadow-lg flex flex-col justify-center items-center cursor-pointer hover:bg-dark-border/50 transition-colors">
                    <span className="text-4xl mb-2">🚀</span>
                    <span className="font-medium text-white">Run Smart Scan</span>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Stock List */}
                <div className="lg:col-span-2 bg-dark-card rounded-xl border border-dark-border shadow-lg overflow-hidden">
                    <div className="p-4 border-b border-dark-border flex justify-between items-center">
                        <h3 className="font-semibold text-white">Top Candidates</h3>
                        <span className="text-xs text-dark-muted">Updated: Just now</span>
                    </div>
                    <StockList onSelect={setSelectedTicker} />
                </div>

                {/* Right: Sniper Detail Card */}
                <div className="lg:col-span-1">
                    <SniperCard ticker={selectedTicker} />
                </div>
            </div>
        </div>
    )
}

export default Dashboard
