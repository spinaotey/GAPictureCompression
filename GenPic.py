import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import stats

def GeneticAlgorithm(picfile,maxIt,nInd,nPol):
    #Initiate population and assess fitness
    population = np.empty(nInd,dtype=np.object)
    fitness = np.empty(nInd,dtype=np.uint64)
    pic = Image.open(picfile)
    bgrgb = stats.mode(np.array(pic.getdata()))[0][0]
    for i in range(nInd):
        population[i] = PicGen(pic,nPol,bgrgb)
        fitness[i] = population[i].getFitness()
    #Find best
    fmax = fitness.argmax()
    best = population[fmax]
    bestFit = fitness[fmax]
    #Main loop
    i = 0   
    for i in range(maxIt):
        P1 = tournamentSelect(population,fitness,i,maxIt)
        P2 = tournamentSelect(population,fitness,i,maxIt)
        C1,C2 = crossover(P1,P2)
        C1.mutate();
        C2.mutate();
        #Substitute two parents with the new children
        ind = np.choice(np.arange(nInd),size=2,replace=False)
        population[ind] = (C1,C2)
        fitness[ind[0]] = population[ind[0]].getFitness()
        if (fitness[ind[0]] > bestFit):
            bestFit = fitness[ind[0]]
            best = population[ind[0]]
        fitness[ind[1]] = population[ind[1]].getFitness()
        if (fitness[ind[1]] > bestFit):
            bestFit = fitness[ind[1]]
            best = population[ind[1]]
    return(best)


def tournamentSelect(population,fitness,i,imax):
    if (np.random.rand() > (0.1+i/imax*0.9)):
        return(np.random.choice(population))
    else:
        ind = np.random.choice(np.arange(fitness.shape[0],replace=False))
#The lower the fitness, the better
        if (fitness[ind[0]] < fitness[ind[1]]):
            return(population[ind[0]])
        else:
            return(population[ind[1]])


def alphaComposite(src, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    src = np.asarray(src)
    dst = np.asarray(dst)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)    
    out[alpha] *= 255
    np.clip(out,0,255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype('uint8')
    out = Image.fromarray(out, 'RGBA')
    return out


def boundedv(x,xmax):
    x[x<0] = 0;
    x[x>xmax] = xmax;

def bounded(x,xmax):
    return(xmax if x > xmax else (0 if x < 0 else x))

class PolyGen:
    bv = 0.001

    def randomPoint(self):
        point = np.empty((1,2),dtype='int32')
        point[0,0] = bounded(np.random.randint(-self.dx,self.xmax+1+self.dx,dtype='int32'),self.xmax)
        point[0,1] = bounded(np.random.randint(-self.dy,self.ymax+1+self.dy,dtype='int32'),self.ymax)
        return(point)

    def __init__(self,x,y,npol=3):
        self.n = np.int8(npol);
        self.coords = np.empty((npol,2),dtype='int32')
        self.xmax = np.int32(x)
        self.ymax = np.int32(y)
        self.dx = x*PolyGen.bv;
        self.coords[:,0] = np.random.randint(-self.dx,x+1+self.dx,npol,dtype='int32')
        boundedv(self.coords[:,0],x)
        self.dy = y*PolyGen.bv;
        self.coords[:,1] = np.random.randint(-self.dy,y+1+self.dy,npol,dtype='int32')
        boundedv(self.coords[:,1],y)
        self.color = np.random.randint(0,256,4,dtype='uint16')

    def mutatePoint1(self,i):
        self.coords[i,:] = self.randomPoint();

    def mutatePoint2(self,i):
        self.coords[i,:] += np.int32(np.random.normal(scale=min(self.xmax,self.ymax)/10,size=2))
        boundedv(self.coords[i,:],self.xmax)

    def mutateColor(self):
        self.color += np.uint16(np.random.normal(scale=256/15,size=4))
        self.color %= 256

    def mutateN(self):
        if self.n == 3:
            i = np.int8(1)
        elif self.n == 6:
            i = np.int8(-1)
        else:
            i = np.random.choice(np.int8([-1,1]))
        self.n += i
        if i == 1:
            self.coords = np.insert(self.coords,
                                    np.random.randint(self.n,size=1,dtype='int8'),
                                    self.randomPoint(),0)
        else:
            self.coords = np.delete(self.coords,
                                    np.random.randint(self.n,size=1,dtype='int8'),
                                    0)
                                    

    def makePoly(self):
        poly = Image.new('RGBA',(self.xmax,self.ymax),None)
        polydraw = ImageDraw.Draw(poly)
        polydraw.polygon(self.coords[:self.n,:].flatten().tolist(),
                         fill=tuple(self.color))
        return(poly)

class PicGen:

    def __init__(self,image,n,bg):
        if(image.mode == "RGB"):
            image.putalpha(255);
        self.trgImg = image
        self.size = image.size
        self.x = image.width;
        self.y = image.height;
        self.n = n;
        self.polygon = np.ndarray((n,),dtype = np.object);
        self.bg = bg
        for i in range(n):
            self.polygon[i] = PolyGen(self.x,self.y,np.random.randint(4)+3)

    def makePic(self):
        self.pic = Image.new("RGBA",self.size,tuple(self.bg))
        for i in range(self.n):
            self.pic = alphaComposite(self.polygon[i].makePoly(),self.pic)
    
    def getFitness(self):
        if not hasattr(self,"pic"):
            self.makePic()
            if hasattr(self,"fitness"):
                del(self.fitness)
        if not hasattr(self,"fitness"):
            self.fitness =  np.sum(np.abs(np.array(list(self.pic.getdata()))-np.array(list(self.trgImg.getdata()))))
        return self.fitness

    def showPic(self):
        if not hasattr(self,"pic"):
            self.makePic();
        self.pic.show();
