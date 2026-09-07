declare module '*/WebThreads' {
  import { FC } from 'react';

  export interface WebThreadsProps {
    color1?: string;
    color2?: string;
    color3?: string;
    speed?: number;
    threadCount?: number;
    frequency?: number;
    spread?: number;
    taper?: number;
    position?: number;
    fanMode?: 'center' | 'left' | 'right';
    glow?: number;
    falloff?: number;
    thickness?: number;
    brightness?: number;
    opacity?: number;
    mirror?: boolean;
    shimmer?: boolean;
    grain?: boolean;
    grainIntensity?: number;
    mouseInteraction?: boolean;
    mouseStrength?: number;
    backgroundColor?: string;
    lightMode?: boolean;
    className?: string;
  }

  const WebThreads: FC<WebThreadsProps>;
  export default WebThreads;
}
