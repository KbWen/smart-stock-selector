import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, LineChart, Activity, Scan, Settings } from 'lucide-react'

interface LayoutProps {
    children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const location = useLocation()

    const navItems = [
        { path: '/', label: 'Dashboard', icon: LayoutDashboard },
        { path: '/backtest', label: 'AI Backtest', icon: LineChart },
        { path: '/risk', label: 'Market Risk', icon: Activity },
        { path: '/indicators', label: 'Scanner', icon: Scan },
    ]

    return (
        <div className="min-h-screen bg-dark-bg text-dark-text font-sans selection:bg-sniper-green selection:text-white flex">
            {/* Sidebar */}
            <aside className="w-64 fixed inset-y-0 left-0 bg-dark-card border-r border-dark-border flex flex-col z-50">
                <div className="h-16 flex items-center px-6 border-b border-dark-border">
                    <span className="text-2xl mr-2">🎯</span>
                    <h1 className="text-lg font-bold tracking-tight text-white">
                        Smart Stock <span className="text-sniper-green align-super text-xs">V4.1</span>
                    </h1>
                </div>

                <nav className="flex-1 px-4 py-6 space-y-1">
                    {navItems.map((item) => {
                        const isActive = location.pathname === item.path
                        const Icon = item.icon
                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${isActive
                                        ? 'bg-sniper-green/10 text-sniper-green'
                                        : 'text-dark-muted hover:bg-dark-border/50 hover:text-white'
                                    }`}
                            >
                                <Icon size={18} />
                                {item.label}
                            </Link>
                        )
                    })}
                </nav>

                <div className="p-4 border-t border-dark-border">
                    <div className="flex items-center gap-3 px-4 py-3 text-dark-muted text-xs">
                        <Settings size={16} />
                        <span>Build: 2026.02.14</span>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 p-8">
                <div className="max-w-7xl mx-auto">
                    {children}
                </div>
            </main>
        </div>
    )
}

export default Layout
