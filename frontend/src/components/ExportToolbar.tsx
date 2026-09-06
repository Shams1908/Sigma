import { motion } from 'framer-motion';

interface ExportToolbarProps {
  analysisId: string | null;
  fileName: string;
  onExportReport: () => void;
  onExportJSON: () => void;
  onExportBits: () => void;
}

export default function ExportToolbar({
  analysisId,
  fileName,
  onExportReport,
  onExportJSON,
  onExportBits
}: ExportToolbarProps) {
  
  const disabled = !analysisId;

  const ExportButton = ({ 
    label, 
    icon, 
    onClick 
  }: { 
    label: string; 
    icon: string; 
    onClick: () => void;
  }) => (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? {} : { scale: 1.02 }}
      whileTap={disabled ? {} : { scale: 0.98 }}
      className={`
        px-4 py-2 rounded-lg font-mono text-xs flex items-center gap-2
        transition-all duration-200
        ${disabled 
          ? 'bg-[#1a1a1a] text-gray-600 cursor-not-allowed border border-[#222222]' 
          : 'bg-[#0A0A0A] text-cyan-400 border border-cyan-900 hover:border-cyan-500 hover:bg-cyan-950/20'
        }
      `}
    >
      <span>{icon}</span>
      <span>{label}</span>
    </motion.button>
  );

  return (
    <div className="bg-[#0A0A0A] rounded-2xl border border-[#222222] p-4 flex items-center justify-between hover:border-cyan-900 transition-colors duration-300">
      <div className="flex items-center gap-4">
        <div className="text-cyan-500 font-mono text-xs tracking-wider uppercase">EXPORT</div>
        {fileName && (
          <div className="text-gray-400 text-xs font-mono">
            {fileName}
          </div>
        )}
      </div>
      
      <div className="flex items-center gap-3">
        <ExportButton 
          label="PDF" 
          icon="📄" 
          onClick={onExportReport} 
        />
        <ExportButton 
          label="JSON" 
          icon="{ }" 
          onClick={onExportJSON} 
        />
        <ExportButton 
          label="BITS" 
          icon="01" 
          onClick={onExportBits} 
        />
      </div>
    </div>
  );
}
