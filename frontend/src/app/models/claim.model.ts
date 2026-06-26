export interface ClaimResult {
  claim: string;
  label: 'supported' | 'contradicted' | 'unsupported';
  evidence: string;
  reasoning: string;
  confidence: 'high' | 'medium' | 'low';
}

export interface VerifyResponse {
  run_id: number;
  status: string;
  error?: string;
  claims?: ClaimResult[];
  stats?: {
    total: number;
    supported: number;
    contradicted: number;
    unsupported: number;
  };
  remaining_quota: number;
}

export interface VerifyRequest {
  answer: string;
  context: string;
  provider: string;
}