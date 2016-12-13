import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import stats
import time

def GeneticAlgorithm(picfile,maxIt,nInd,nPol,nNew,nTour,p,colScale=100,coordProp=10):
    #Initiate population and assess fitness
    PolyGen.colScale = colScale
    PolyGen.coordProp = coordProcoordProp
    population = np.empty(nInd,dtype=np.object)
    fitness = np.empty(nInd,dtype=np.uint64)
    pic = Image.open(picfile)
    bgrgb = stats.mode(np.array(pic.getdata()))[0][0]
    for i in range(nInd):
        population[i] = PicGen.new(pic,nPol,bgrgb)
        fitness[i] = population[i].getFitness()
    #Find best
    fmin = fitness.argmin()
    best = population[fmin]
    bestFit = fitness[fmin]
    #Main loop
    bests = np.empty(maxIt+1,dtype=np.uint64)
    bests[0] = bestFit
    for i in range(maxIt):
        t = time.time()
        C = np.empty(nNew,dtype=np.object)
        cFitness = np.empty(nNew,dtype=np.uint64)
        for j in range(nNew//2):
            P1 = tournamentSelect(population,fitness,nTour)
            P2 = tournamentSelect(population,fitness,nTour)
            C[j*2],C[j*2+1] = crossover(P1,P2)
            C[j*2].mutate(p);
            C[j*2+1].mutate(p);
            cFitness[j*2] = C[j*2].getFitness()
            cFitness[j*2+1] = C[j*2+1].getFitness()
        #Substitute parents with the new children
        ind = np.random.choice(np.arange(nInd),size=nNew,replace=False)
        population[ind] = C
        fitness[ind] = cFitness
        if cFitness.min() < bestFit:
            best = C[cFitness.argmin()]
        bests[i+1] = fitness.min()
        print('Iteration %d: %lf s.\n'%(i+1,time.time()-t))
    return(best,bests)


def tournamentSelect(population,fitness,nTour):
    ind = np.random.choice(np.arange(fitness.shape[0]),size=nTour)
    m = fitness[ind].argmin()
    return(population[ind[m]])
    
def crossover(P1,P2):
    C1 = PicGen(P1.trgImg,P1.n,P1.bg)
    C2 = PicGen(P1.trgImg,P1.n,P1.bg)
    for i in range(0,P1.n,2):
        C1.polygon[i] = P1.polygon[i]
        C2.polygon[i] = P2.polygon[i]
    for i in range(1,P1.n,2):
        C1.polygon[i] = P2.polygon[i]
        C2.polygon[i] = P1.polygon[i]
    return(C1,C2)



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
    colScale = 100
    coordProp = 10

    def randomPoint(self):
        point = np.empty((1,2),dtype='int32')
        point[0,0] = bounded(np.random.randint(-self.dx,self.x+1+self.dx,dtype='int32'),self.x)
        point[0,1] = bounded(np.random.randint(-self.dy,self.y+1+self.dy,dtype='int32'),self.y)
        return(point)


    def __init__(self,x=0,y=0,n=3):
        self.n = np.int8(n)
        self.x = np.int32(x)
        self.y = np.int32(y)
        self.dx = x*PolyGen.bv
        self.dy = y*PolyGen.bv
        self.coords = np.empty((n,2),dtype='int32')

    def new(x,y,n):
        new = PolyGen(x,y,n)
        new.coords[:,0] = np.random.randint(-new.dx,x+1+new.dx,n,dtype='int32')
        boundedv(new.coords[:,0],new.x)
        new.coords[:,1] = np.random.randint(-new.dy,y+1+new.dy,n,dtype='int32')
        boundedv(new.coords[:,1],y)
        new.color = np.int16(np.random.normal(scale=10,size=4))
        boundedv(new.color,255)
        return(new)

    def mutatePoint1(self,i):
        self.coords[i,:] = self.randomPoint();

    def mutatePoint2(self,i):
        self.coords[i,:] += np.int32(np.random.normal(scale=min(self.x,self.y)/PolyGen.coordProp,size=2))
        self.coords[i,0] = bounded(self.coords[i,0],self.x)
        self.coords[i,1] = bounded(self.coords[i,1],self.y)
        if hasattr(self,"poly"):
            del(self.poly)

    def mutateColor(self):
        self.color += np.int16(np.random.normal(scale=PolyGen.colScale,size=4))
        boundedv(self.color,255)
        if hasattr(self,"poly"):
            del(self.poly)

    def mutateN(self):
        if hasattr(self,"poly"):
            del(self.poly)
        if self.n == 3:
            i = np.int8(1)
        elif self.n == 6:
            i = np.int8(-1)
        else:
            i = np.random.choice(np.int8([-1,1]))
        self.n += i
        if i == 1:
            j = np.random.randint(self.n,size=1,dtype='int8')
            self.coords = np.insert(self.coords,j,self.coords.mean(axis=0),0)
            self.mutatePoint2(j)
        else:
            self.coords = np.delete(self.coords,
                                    np.random.randint(self.n,size=1,dtype='int8')
                                    ,0)
                                    

    def makePoly(self):
        if not hasattr(self,"poly"):
            self.poly = Image.new('RGBA',(self.x,self.y),None)
            polydraw = ImageDraw.Draw(self.poly)
            polydraw.polygon(self.coords[:self.n,:].flatten().tolist(),
                             fill=tuple(self.color))
        return(self.poly)

class PicGen:

    def __init__(self,image,n,bg):
        if(image.mode == "RGB"):
            image.putalpha(255);
        self.trgImg = image
        self.size = image.size
        self.x = image.width;
        self.y = image.height;
        self.n = n;
        self.polygon = np.ndarray(n,dtype = np.object);
        self.bg = bg

    def new(image,n,bg):
        new = PicGen(image,n,bg)
        for i in range(n):
            new.polygon[i] = PolyGen.new(new.x,new.y,np.random.randint(4)+3)
        return(new)


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

    def mutate(self,p):
#        for i in np.argwhere(np.random.rand(self.n)<(p/10)).flatten():
#            self.polygon[i].mutateN()
        i = -1
        for i in np.argwhere(np.random.rand(self.n)<p).flatten():
            self.polygon[i].mutateColor()
        if i != -1:
            if hasattr(self,"pic"):
                del(self.pic)
        j = -1
        for i in np.argwhere(np.random.rand(self.n)<p).flatten():
            for j in np.argwhere(np.random.rand(self.polygon[i].n)<0.5):
                self.polygon[i].mutatePoint2(j)
        if j != -1:
            if hasattr(self,"pic"):
                del(self.pic)
