export interface ClaimResult {
  claim: string;
  label: 'supported' | 'contradicted' | 'unsupported';
  evidence: string;
  reasoning: string;
  confidence: 'high' | 'medium' | 'low';
}

export interface VerifyResponse {
  claims: ClaimResult[];
  stats: {
    total: number;
    supported: number;
    contradicted: number;
    unsupported: number;
  };
}

export interface VerifyRequest {
  answer: string;
  context: string;
  provider: string;
}