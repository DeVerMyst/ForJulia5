using NFFT

nx = 64
nb = 195
nw = 20

tab_u = rand(nb,nw) - .5
tab_v = rand(nb,nw) - .5

X = randn(nx,nx,nw) + 0.*im

Fx = SharedArray( Complex{Float64}, (nb, nw))


Plan = Dict{}()
for n in 1:nw
  Plan[n] = NFFTPlan( hcat(tab_u[:,n],tab_v[:,n])', (nx,nx))
end

@sync @parallel for n in 1:nw
  println("worker")
    Fx[:,n]  = nfft(Plan[n], X[:,:,n])
end
