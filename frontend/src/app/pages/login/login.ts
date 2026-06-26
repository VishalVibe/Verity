import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ToastService } from '../../services/toast.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.html',
})
export class LoginComponent {
  email = '';
  password = '';
  loading = false;
  error = '';

  constructor(
    private authService: AuthService,
    private router: Router,
    private cdr: ChangeDetectorRef,
    private toastService: ToastService
  ) {}

  submit() {
    if (!this.email || !this.password) return;
    this.loading = true;
    this.error = '';

    this.authService.login(this.email, this.password).subscribe({
      next: () => {
        this.toastService.show('Logged in successfully!', 'success');
        this.router.navigate(['/']);
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Invalid email or password';
        this.toastService.show(this.error, 'error');
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }
}