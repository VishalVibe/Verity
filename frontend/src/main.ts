import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppShell } from './app/app-shell';

bootstrapApplication(AppShell, appConfig).catch(console.error);