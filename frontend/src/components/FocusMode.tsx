import { ReactNode, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface FocusModeProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

export default function FocusMode({ isOpen, onClose, title, children }: FocusModeProps) {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#050505]"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="relative w-[95vw] h-[95vh] bg-[#0A0A0A] border border-[#222222] rounded-2xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#222222]">
              <div className="flex items-center gap-4">
                <h2 className="text-xl font-bold text-white uppercase tracking-wide">{title}</h2>
                <span className="text-xs font-mono text-gray-500">FOCUS MODE</span>
              </div>
              <button
                onClick={onClose}
                className="px-4 py-2 bg-[#111111] hover:bg-[#1a1a1a] border border-[#222222] hover:border-red-500 text-white text-sm font-semibold rounded-lg transition-all"
              >
                Close (ESC)
              </button>
            </div>
            <div className="flex-1 overflow-hidden p-6">
              {children}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
