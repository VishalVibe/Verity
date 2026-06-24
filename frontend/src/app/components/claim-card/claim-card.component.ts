import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ClaimResult } from '../../models/claim.model';

@Component({
  selector: 'app-claim-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div
      class="bg-white rounded-lg border-l-4 border border-gray-100 p-3 cursor-pointer select-none transition-all"
      [ngClass]="borderColor"
      (click)="expanded = !expanded">

      <div class="flex justify-between items-start gap-3">
        <p class="text-sm text-gray-800 leading-relaxed flex-1">{{ claim.claim }}</p>
        <div class="flex items-center gap-2 flex-shrink-0">
          <span class="text-xs font-semibold uppercase tracking-wide px-2 py-0.5 rounded"
            [ngClass]="badgeClass">
            {{ claim.label }}
          </span>
          <span class="text-gray-400 text-xs">{{ expanded ? '▲' : '▼' }}</span>
        </div>
      </div>

      <div *ngIf="expanded" class="mt-3 pt-3 border-t border-gray-100 space-y-2">
        <div>
          <p class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Evidence</p>
          <p class="text-xs text-gray-600 leading-relaxed font-mono bg-gray-50 rounded p-2">
            {{ claim.evidence || 'No matching evidence found in source.' }}
          </p>
        </div>
        <div>
          <p class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Reasoning</p>
          <p class="text-xs text-gray-600 leading-relaxed">{{ claim.reasoning }}</p>
        </div>
        <div class="flex items-center gap-1.5">
          <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</p>
          <span class="text-xs px-2 py-0.5 rounded-full font-medium"
            [ngClass]="confidenceClass">
            {{ claim.confidence }}
          </span>
        </div>
      </div>
    </div>
  `
})
export class ClaimCardComponent {
  @Input() claim!: ClaimResult;
  expanded = false;

  get borderColor(): string {
    return {
      supported: 'border-l-supported-border',
      contradicted: 'border-l-contradicted-border',
      unsupported: 'border-l-unsupported-border',
    }[this.claim.label];
  }

  get badgeClass(): string {
    return {
      supported: 'bg-supported-bg text-supported-text',
      contradicted: 'bg-contradicted-bg text-contradicted-text',
      unsupported: 'bg-unsupported-bg text-unsupported-text',
    }[this.claim.label];
  }

  get confidenceClass(): string {
    return {
      high: 'bg-green-100 text-green-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-gray-100 text-gray-600',
    }[this.claim.confidence];
  }
}