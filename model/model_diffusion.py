import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def cosine_similarity(X,Y):
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)

    corr = norm(X) * norm(Y)
    similarity = corr.sum(dim=0).mean(dim=1)

    return similarity


def norm(t):
    return F.normalize(t,dim=1,eps=1e-10)


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,model,beta_l,beta_T,T):
        super().__init__()
        self.model = model
        self.T=T

        self.register_buffer(
            'beats',torch.linspace(beta_l,beta_T,T).double())
        
        alphas = 1.-self.betas
        alphas_bar = torch.cumprod(alphas,dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
    def forward(self,x_0,condition):
        t = torch.randint(self.T,size=(x_0.shape[0],),device=x_0.device)
        noise = torch.randn_like(x_0)

        x_t = (
            extract(self.sqrt_alphas_bar,t,x_0.shape)*x_0+
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)*noise
        )

        x_recon = self.model(torch.cat([x_t,condition],dim=1),t)

        loss_mse = F.mse_loss(x_recon,noise,reduction='none').mean()

        return loss_mse


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double()
        )

        alphas = 1.-self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        log_var = extract(self.posterior_log_var_clipped, t, x_t.shape)
        return mean, log_var
    
    def p_mean_variance(self, x_t, condition, t):
        model_out = self.model(torch.cat([x_t, condition], dim=1), t)

        if self.mean_type == 'eps':
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=model_out)
        elif self.mean_type == 'xstart':
            x_0 = model_out
        else:
            raise ValueError(f"invalid mean type: {self.mean_type}")
        
        x_0 = torch.clamp(x_0, -1., 1.)
        mean, log_var = self.q_mean_variance(x_0=x_0, x_t=x_t, t=t)
        
        return mean, log_var
    
    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps)   
     
    def forward(self, x_T, condition):
        out = []
        x_t = x_T 
        
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, condition=condition, t=t)
            
            if time_step > 0:
                noise = torch.randn_like(x_t)  
            else:
                noise = 0.
            
            x_t = mean + torch.exp(0.5 * log_var) * noise
            out.append(x_t)
        
        return out