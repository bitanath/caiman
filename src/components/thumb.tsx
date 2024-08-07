import { useEffect, useState, useRef } from "react";

export const useIntersection = (element, rootMargin) => {
  const [isVisible, setState] = useState(false);

  useEffect(() => {
    const current = element?.current;
    const observer = new IntersectionObserver(
      ([entry]) => {
        setState(entry.isIntersecting);
      },
      { rootMargin }
    );
    current && observer?.observe(current);

    return () => current && observer.unobserve(current);
  }, []);

  return isVisible;
};

export interface ImageProps{
    thumbnailUrl: string,
    heatmapUrl: string,
    alternateUrl: string,
    prompt: string,
    message: string
}

export const DummyElement = ({
  onVisible,
  key,
  imageProps
}: {
  onVisible: () => {};
  key: number;
  imageProps: ImageProps;
}) => {
  const triggerRef = useRef(null);
  const isVisible = useIntersection(triggerRef, "0px");

  useEffect(() => {
    if (isVisible) {
      onVisible(); // Trigger a function when the div is visible on view port
    }
  }, [onVisible, isVisible]);


  return <div ref={triggerRef}></div>;
};
