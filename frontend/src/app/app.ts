import { Component, ChangeDetectorRef, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VerifyService } from './services/verify.service';
import { VerifyResponse } from './models/claim.model';
import { ClaimCardComponent } from './components/claim-card/claim-card.component';
import { Router } from '@angular/router';
import { AuthService } from './services/auth.service';
import { ToastService, Toast } from './services/toast.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ClaimCardComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  answer = '';
  context = '';
  provider = 'groq';
  loading = false;
  error = '';
  response: VerifyResponse | null = null;

  // UX additions
  toasts: Toast[] = [];
  private toastSub: any = null;

  // Custom Loading states for premium skeletons
  dashboardLoading = false;
  historyLoading = false;
  apiKeysLoading = false;

  // View toggles: dashboard | verify | history | detail | api-keys
  viewMode: 'dashboard' | 'verify' | 'history' | 'detail' | 'api-keys' = 'dashboard';
  pastRuns: any[] = [];
  selectedRun: any = null;
  quotaRemaining: number | null = null;

  // Dashboard Stats
  dashboardStats: any = null;

  // API Key Management
  apiKeys: any[] = [];
  newKeyName = '';
  generatedApiKey = '';

  // Background Job Polling
  pollInterval: any = null;
  pollingRunId: number | null = null;
  statusMessage = '';

  constructor(
    private verifyService: VerifyService,
    private authService: AuthService,
    private router: Router,
    private cdr: ChangeDetectorRef,
    private toastService: ToastService
  ) {}

  ngOnInit() {
    this.loadUserQuota();
    this.loadDashboardStats();
    this.loadPastRuns();

    this.toastSub = this.toastService.toasts$.subscribe(t => {
      this.toasts = t;
      this.cdr.detectChanges();
    });
  }

  ngOnDestroy() {
    this.stopPolling();
    if (this.toastSub) {
      this.toastSub.unsubscribe();
    }
  }

  loadUserQuota() {
    this.authService.me().subscribe({
      next: (user) => {
        this.quotaRemaining = user.remaining_quota;
        this.cdr.detectChanges();
      },
      error: () => {}
    });
  }

  loadPastRuns() {
    this.historyLoading = true;
    this.verifyService.getRuns().subscribe({
      next: (runs) => {
        this.pastRuns = runs;
        this.historyLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.historyLoading = false;
        console.error('Failed to load past runs:', err);
      }
    });
  }

  loadDashboardStats() {
    this.dashboardLoading = true;
    this.verifyService.getDashboardStats().subscribe({
      next: (stats) => {
        this.dashboardStats = stats;
        this.dashboardLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.dashboardLoading = false;
        console.error('Failed to load dashboard stats:', err);
      }
    });
  }

  loadApiKeys() {
    this.apiKeysLoading = true;
    this.verifyService.listApiKeys().subscribe({
      next: (keys) => {
        this.apiKeys = keys;
        this.apiKeysLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.apiKeysLoading = false;
        console.error('Failed to load API keys:', err);
      }
    });
  }

  createKey() {
    if (!this.newKeyName.trim()) return;
    this.verifyService.createApiKey(this.newKeyName).subscribe({
      next: (res) => {
        this.generatedApiKey = res.api_key;
        this.newKeyName = '';
        this.loadApiKeys();
        this.toastService.show('API Key generated successfully!', 'success');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.toastService.show('Failed to generate API Key.', 'error');
      }
    });
  }

  revokeKey(id: number) {
    this.verifyService.revokeApiKey(id).subscribe({
      next: () => {
        this.loadApiKeys();
        this.toastService.show('API Key revoked successfully.', 'info');
      },
      error: (err) => {
        this.toastService.show('Failed to revoke API Key.', 'error');
      }
    });
  }

  copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      this.toastService.show('API Key copied to clipboard!', 'success');
    }).catch(err => {
      this.toastService.show('Failed to copy API key.', 'error');
    });
  }

  setViewMode(mode: 'dashboard' | 'verify' | 'history' | 'detail' | 'api-keys') {
    this.viewMode = mode;
    this.error = '';
    this.stopPolling();
    this.loading = false;
    
    if (mode === 'dashboard') {
      this.loadDashboardStats();
    } else if (mode === 'history') {
      this.loadPastRuns();
    } else if (mode === 'api-keys') {
      this.loadApiKeys();
      this.generatedApiKey = '';
    }
    this.cdr.detectChanges();
  }

  viewRunDetail(runId: number) {
    this.loading = true;
    this.error = '';
    this.cdr.detectChanges();

    this.verifyService.getRun(runId).subscribe({
      next: (run) => {
        this.selectedRun = run;
        this.viewMode = 'detail';
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Failed to load verification run details.';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  getDownloadUrl(runId: number, format: 'json' | 'pdf'): string {
    if (format === 'pdf') {
      return `http://localhost:8000/runs/${runId}/pdf`;
    }
    return `http://localhost:8000/runs/${runId}`;
  }

  downloadJson(run: any) {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(run, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `verity_report_${run.id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }

  logout() {
    this.authService.logout();
    this.toastService.show('Logged out successfully.', 'info');
    this.router.navigate(['/login']);
  }

  startPolling(runId: number) {
    this.stopPolling();
    this.statusMessage = 'Job submitted to background queue (waiting)...';
    this.cdr.detectChanges();

    this.pollInterval = setInterval(() => {
      this.verifyService.getRun(runId).subscribe({
        next: (run) => {
          if (run.status === 'completed') {
            this.stopPolling();
            this.response = {
              run_id: run.id,
              status: run.status,
              claims: run.claims,
              stats: run.stats,
              remaining_quota: this.quotaRemaining || 0
            };
            this.loading = false;
            this.toastService.show('Claims verification completed!', 'success');
            this.loadPastRuns();
            this.loadDashboardStats();
            this.cdr.detectChanges();
          } else if (run.status === 'failed') {
            this.stopPolling();
            this.error = run.error || 'Async claim verification task failed.';
            this.toastService.show(this.error, 'error');
            this.loading = false;
            this.cdr.detectChanges();
          } else {
            this.statusMessage = `Task processing status: ${run.status}...`;
            this.cdr.detectChanges();
          }
        },
        error: (err) => {
          this.stopPolling();
          this.error = 'Failed to poll background task status.';
          this.toastService.show(this.error, 'error');
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
    }, 1500);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.pollingRunId = null;
  }

  verify() {
    if (!this.answer.trim() || !this.context.trim()) return;
    this.loading = true;
    this.error = '';
    this.response = null;
    this.cdr.detectChanges();

    this.verifyService.verify({
      answer: this.answer,
      context: this.context,
      provider: this.provider,
    }).subscribe({
      next: (res) => {
        this.quotaRemaining = res.remaining_quota;
        if (res.status === 'completed') {
          this.response = res;
          this.loading = false;
          this.toastService.show('Claims verification completed!', 'success');
          this.loadPastRuns();
          this.loadDashboardStats();
          this.cdr.detectChanges();
        } else {
          this.pollingRunId = res.run_id;
          this.startPolling(res.run_id);
        }
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Something went wrong. Is the backend running?';
        this.toastService.show(this.error, 'error');
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  get activityHistoryPoints(): string {
    if (!this.dashboardStats || !this.dashboardStats.activity_history || !this.dashboardStats.activity_history.length) return '';
    const points = this.dashboardStats.activity_history;
    const count = points.length;
    return points.map((pt: any, index: number) => {
      const x = count > 1 ? (index / (count - 1)) * 500 : 250;
      const y = 90 - (pt.accuracy / 100) * 80;
      return `${x},${y}`;
    }).join(' ');
  }

  get activityHistoryDots(): any[] {
    if (!this.dashboardStats || !this.dashboardStats.activity_history || !this.dashboardStats.activity_history.length) return [];
    const points = this.dashboardStats.activity_history;
    const count = points.length;
    return points.map((pt: any, index: number) => {
      const x = count > 1 ? (index / (count - 1)) * 500 : 250;
      const y = 90 - (pt.accuracy / 100) * 80;
      return { x, y };
    });
  }

  getBreakdownPercent(type: string): number {
    if (!this.dashboardStats || !this.dashboardStats.hallucinations_breakdown) return 0;
    const b = this.dashboardStats.hallucinations_breakdown;
    const total = (b.supported || 0) + (b.contradicted || 0) + (b.unsupported || 0);
    if (total === 0) return 0;
    const count = b[type] || 0;
    return (count / total) * 100;
  }
}