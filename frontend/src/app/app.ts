import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VerifyService } from './services/verify.service';
import { VerifyResponse } from './models/claim.model';
import { ClaimCardComponent } from './components/claim-card/claim-card.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ClaimCardComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  answer = '';
  context = '';
  provider = 'groq';
  loading = false;
  error = '';
  response: VerifyResponse | null = null;

  constructor(
    private verifyService: VerifyService,
    private cdr: ChangeDetectorRef
  ) {}

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
        this.response = res;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Something went wrong. Is the backend running?';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }
}