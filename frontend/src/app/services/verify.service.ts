import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { VerifyRequest, VerifyResponse } from '../models/claim.model';

@Injectable({ providedIn: 'root' })
export class VerifyService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  verify(payload: VerifyRequest): Observable<VerifyResponse> {
    return this.http.post<VerifyResponse>(`${this.apiUrl}/verify`, payload);
  }

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.apiUrl}/health`);
  }
}