import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface PremiumCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  badge?: string;
  delay?: number;
  enableHover?: boolean;
}

export default function PremiumCard({ 
  children, 
  className = '', 
  title, 
  badge,
  delay = 0,
  enableHover = true
}: PremiumCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      whileHover={enableHover ? { y: -5 } : undefined}
      className={`
        bg-[#0A0A0A] 
        border border-[#222222] 
        hover:border-sigma-teal-900 
        rounded-2xl 
        p-6 
        transition-colors 
        duration-300
        relative
        overflow-hidden
        ${className}
      `.trim()}
    >
      {/* Optional colored glow effect in corner */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-sigma-teal/5 rounded-full blur-[60px] pointer-events-none" />
      
      {/* Content container with relative positioning */}
      <div className="relative z-10">
        {/* Header section */}
        {(title || badge) && (
          <div className="mb-4 space-y-2">
            {badge && (
              <div className="inline-flex items-center gap-2 bg-sigma-teal/10 border border-sigma-teal/30 rounded-full px-3 py-1">
                <div className="w-1.5 h-1.5 bg-sigma-teal rounded-full animate-pulse" />
                <span className="text-sigma-teal text-xs font-mono font-bold tracking-wider uppercase">
                  {badge}
                </span>
              </div>
            )}
            
            {title && (
              <div className="text-sigma-teal font-mono text-xs tracking-wider uppercase">
                {title}
              </div>
            )}
          </div>
        )}
        
        {/* Main content */}
        {children}
      </div>
      
      {/* Bottom accent line that grows on hover */}
      <motion.div 
        className="absolute bottom-0 left-0 h-[2px] w-0 bg-gradient-to-r from-sigma-teal to-sigma-purple group-hover:w-full transition-all duration-500"
        initial={{ width: 0 }}
        whileHover={{ width: '100%' }}
      />
    </motion.div>
  );
}
