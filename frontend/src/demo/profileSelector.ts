/**
 * profileSelector.ts — Deterministic file → demo profile mapping.
 *
 * IMPLEMENTATION NOTE (for developers):
 * This module does NOT perform real signal classification.
 * It selects a presentation profile purely from upload metadata
 * (filename + file size) that is already available in the browser.
 *
 * Selection priority:
 *   1. Keyword match in the filename (case-insensitive)
 *   2. djb2 hash of (filename + file size) mapped to one of the 5 profiles
 *
 * The same file will always produce the same profile.
 * Different files will usually produce different profiles.
 *
 * Called from WorkstationNew.tsx immediately after a file is selected,
 * before the upload starts.  The profile is stored in component state
 * and used only where real backend results are absent.
 */

import { ALL_PROFILES, PROFILES_BY_ID, type DemoProfile } from './demoProfiles';

// ---------------------------------------------------------------------------
// djb2 hash — deterministic, no dependencies, browser-safe
// ---------------------------------------------------------------------------
function djb2(s: string): number {
  let hash = 5381;
  for (let i = 0; i < s.length; i++) {
    // hash * 33 XOR charCode — standard djb2 variant
    hash = (hash * 33) ^ s.charCodeAt(i);
  }
  // Force to unsigned 32-bit integer
  return hash >>> 0;
}

// ---------------------------------------------------------------------------
// Keyword table — checked in order; first match wins
// ---------------------------------------------------------------------------
const KEYWORD_MAP: Array<{ keywords: string[]; profileId: string }> = [
  { keywords: ['16qam', '16-qam', 'qam16', 'qam-16'],   profileId: '16qam' },
  { keywords: ['8psk',  '8-psk',  'psk8',  'psk-8'],    profileId: '8psk'  },
  { keywords: ['qpsk',  'qpsk4',  '4psk'],               profileId: 'qpsk'  },
  { keywords: ['bpsk',  'bpsk1',  'dpsk'],               profileId: 'bpsk'  },
  { keywords: ['4fsk',  '4-fsk',  'fsk4'],               profileId: 'fsk'   },
  { keywords: ['2fsk',  '2-fsk',  'fsk2', 'fsk', 'gfsk', 'mfsk'], profileId: 'fsk' },
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Select a DemoProfile deterministically from the file object.
 *
 * @param file  The File object from the upload input (metadata only — bytes are never read)
 * @returns     A DemoProfile to use as fallback data for this file
 */
export function selectDemoProfile(file: File): DemoProfile {
  const lower = file.name.toLowerCase();

  // 1. Keyword match
  for (const entry of KEYWORD_MAP) {
    if (entry.keywords.some(kw => lower.includes(kw))) {
      return PROFILES_BY_ID[entry.profileId];
    }
  }

  // 2. Hash fallback — use filename + size so different files of the same
  //    name but different sizes still resolve differently.
  const key  = `${file.name}::${file.size}`;
  const hash = djb2(key);
  const idx  = hash % ALL_PROFILES.length;
  return ALL_PROFILES[idx];
}

/**
 * Convenience: select profile from filename + size strings.
 * Useful for unit-testing without a real File object.
 */
export function selectDemoProfileByMeta(name: string, size: number): DemoProfile {
  const lower = name.toLowerCase();

  for (const entry of KEYWORD_MAP) {
    if (entry.keywords.some(kw => lower.includes(kw))) {
      return PROFILES_BY_ID[entry.profileId];
    }
  }

  const key  = `${name}::${size}`;
  const hash = djb2(key);
  const idx  = hash % ALL_PROFILES.length;
  return ALL_PROFILES[idx];
}
