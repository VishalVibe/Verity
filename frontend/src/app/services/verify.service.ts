import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { VerifyRequest, VerifyResponse } from '../models/claim.model';

@Injectable({ providedIn: 'root' })
export class VerifyService {
  private apiUrl = 'https://verity-api-cz28.onrender.com';

  constructor(private http: HttpClient) {}

  verify(payload: VerifyRequest): Observable<VerifyResponse> {
    return this.http.post<VerifyResponse>(`${this.apiUrl}/verify`, payload);
  }

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.apiUrl}/health`);
  }

  getRuns(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/runs`);
  }

  getRun(runId: number): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/runs/${runId}`);
  }

  getDashboardStats(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/dashboard/stats`);
  }

  listApiKeys(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/auth/keys`);
  }

  createApiKey(name: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/auth/keys`, { name });
  }

  revokeApiKey(id: number): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/auth/keys/${id}`);
  }
}