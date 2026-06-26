import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ToastService } from '../../services/toast.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './register.html',
})
export class RegisterComponent {
  email = '';
  username = '';
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
    if (!this.email || !this.username || !this.password) return;
    this.loading = true;
    this.error = '';

    this.authService.register(this.email, this.username, this.password).subscribe({
      next: () => {
        this.toastService.show('Account created successfully!', 'success');
        this.router.navigate(['/']);
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Registration failed. Try again.';
        this.toastService.show(this.error, 'error');
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }
}