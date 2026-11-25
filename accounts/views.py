from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import SignupForm, LoginForm

def signup_view(request):
    if request.user.is_authenticated: return redirect('/chat/')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.first_name or user.username}!')
            return redirect('/chat/')
        for f, errs in form.errors.items():
            for e in errs: messages.error(request, f'{f}: {e}')
    else:
        form = SignupForm()
    return render(request, 'accounts/signup.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated: return redirect('/chat/')
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(request, username=form.cleaned_data['username'], password=form.cleaned_data['password'])
            if user:
                login(request, user)
                return redirect(request.GET.get('next', '/chat/'))
            messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('/accounts/login/')

@login_required
def profile_view(request):
    from chat.models import ChatSession
    return render(request, 'accounts/profile.html', {
        'user': request.user,
        'session_count': ChatSession.objects.filter(user=request.user).count(),
    })
